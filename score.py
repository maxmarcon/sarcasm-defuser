from sentence_transformers import SentenceTransformer, SimilarityFunction
from transformers import pipeline
import itertools, argparse, logging, os, flatdict
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from typing import Iterable
from datasets import Dataset
from torch.utils.data import DataLoader
from huggingface_hub import metadata_update, EvalResult, login
import torch
from modelbag import ModelBag

CONFIG_KEYS = ["generate_params", "sarcasm_model", "similarity_model", "model"]
METRIC_KEYS = ["similarity", "sarcasm_prob_orig", "sarcasm_prob_neutral"]
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
QUARTILES = [0.25, 0.5, 0.75]

logger = logging.getLogger(__name__)


def make_stats(vals):
    quartiles = np.quantile(vals, QUARTILES).tolist()
    return {
        **{
            "mean": np.mean(vals).item(),
            "std": np.std(vals).item(),
            "min": np.min(vals).item(),
            "max": np.max(vals).item(),
        },
        **{f"q{i+1}": quartiles[i] for i in range(len(quartiles))},
    }


class SarcasmScorer:

    DEFAULT_SIMILARITY_MODEL = "multi-qa-mpnet-base-dot-v1"
    DEFAULT_SARCASM_MODEL = "helinivan/english-sarcasm-detector"

    def __init__(
        self,
        *,
        similarity_model_name=DEFAULT_SIMILARITY_MODEL,
        sarcasm_model_name=DEFAULT_SARCASM_MODEL,
        sarcasm_label="LABEL_1",
        batch_size=10,
    ):
        self.similarity_model_name = similarity_model_name
        self.similarity_model = SentenceTransformer(
            similarity_model_name,
            similarity_fn_name=SimilarityFunction.COSINE,
            device=DEVICE,
        )

        self.sarcasm_model_name = sarcasm_model_name
        self.sarcasm_pipe = pipeline(
            "text-classification", model=sarcasm_model_name, device=DEVICE
        )
        assert (
            self.sarcasm_pipe.model.config.id2label[1] == sarcasm_label
        ), f"sarcasm_label '{sarcasm_label}' not in model's labels"

        self.sarcasm_label = sarcasm_label
        self.batch_size = batch_size

    def _get_sarc_prob(self, x):
        return x["score"] if x["label"] == self.sarcasm_label else 1.0 - x["score"]

    def score(
        self,
        original_sarcastic: Iterable[str],
        generated_neutral: Iterable[str],
        *,
        total: int = None,
        verbose: bool = False,
    ) -> dict[str, str | float | dict[str, float]]:
        original_sarcastic_len = (
            len(original_sarcastic) if hasattr(original_sarcastic, "__len__") else None
        )
        generated_neutral_len = (
            len(generated_neutral) if hasattr(generated_neutral, "__len__") else None
        )

        if (
            original_sarcastic_len is not None
            and generated_neutral_len is not None
            and original_sarcastic_len != generated_neutral_len
        ):
            raise ValueError(
                f"original_sarcastic and generated_neutral must be same length, bug got {original_sarcastic_len} and {generated_neutral_len}"
            )

        if original_sarcastic_len is not None:
            total = original_sarcastic_len
        elif generated_neutral_len is not None:
            total = generated_neutral_len

        similarity = []
        sarc_prob_gen = []
        sarc_prob_orig = []
        n = 0
        total_steps = None if total is None else total // self.batch_size

        for orig, gen in tqdm(
            zip(
                itertools.batched(original_sarcastic, self.batch_size),
                itertools.batched(generated_neutral, self.batch_size),
            ),
            desc="Computing similarity and sarcasm scores",
            total=total_steps,
        ):
            if len(orig) != len(gen):
                raise ValueError(
                    f"got different sizes in batch fro orig and gen: {len(orig)} and {len(gen)}"
                )

            n += len(gen)

            if verbose:
                for o, g in zip(orig, gen):
                    print(f"Original:\n{o}")
                    print(f"Generated:\n{g}")

            emb1 = self.similarity_model.encode(
                orig, convert_to_tensor=True, show_progress_bar=False
            )
            emb2 = self.similarity_model.encode(
                gen, convert_to_tensor=True, show_progress_bar=False
            )

            sims = self.similarity_model.similarity_pairwise(emb1, emb2)
            similarity += [x.item() for x in sims]

            gen_sarcs = self.sarcasm_pipe(list(gen), truncation=True)
            orig_sarcs = self.sarcasm_pipe(list(orig), truncation=True)

            for gen_sarc, orig_sarc in zip(gen_sarcs, orig_sarcs):
                sarc_prob_gen.append(self._get_sarc_prob(gen_sarc))
                sarc_prob_orig.append(self._get_sarc_prob(orig_sarc))

        if total is not None:
            assert total == n, f"was passed total of {total} but found {n} elements"

        return {
            "sarcasm_model": self.sarcasm_model_name,
            "similarity_model": self.similarity_model_name,
            "similarity": make_stats(similarity),
            "sarcasm_prob_orig": make_stats(sarc_prob_orig),
            "sarcasm_prob_neutral": make_stats(sarc_prob_gen),
        }


def write_results(results: list[dict], results_file: str):
    pd.DataFrame([flatdict.FlatDict(result) for result in results]).to_csv(
        results_file, index=False
    )


def score_model(
    model_name: str,
    scorer: SarcasmScorer,
    test_data: pd.DataFrame,
    generate_params: list[dict],
    *,
    compute_baseline=False,
    results_file=None,
    verbose=False,
) -> dict:
    if compute_baseline:
        logger.info("Computing baseline score...")
        baseline_score = scorer.score(
            test_data["comment"], test_data["neutral_comment"]
        )
        results = [{**baseline_score, "model": "baseline"}]
        write_results(results, results_file)
        return results

    mb = ModelBag(model_name, for_training=False, device=DEVICE)

    ds = Dataset.from_pandas(test_data)
    ds = ds.map(mb.preprocessor, batched=True, remove_columns=ds.column_names)
    data_loader = DataLoader(ds, collate_fn=mb.data_collator, batch_size=8)

    model_results = []

    for params in generate_params:
        logger.info(f"scoring with generate_params = {params}")

        generated = (
            mb.model.generate(
                **{
                    **{k: v.to(DEVICE) for k, v in batch.items()},
                    **params,
                },
                pad_token_id=mb.tokenizer.pad_token_id,
                bos_token_id=mb.tokenizer.bos_token_id,
                eos_token_id=mb.tokenizer.eos_token_id,
            )
            for batch in data_loader
        )

        generated_decoded = (
            mb.tokenizer.batch_decode(batch, skip_special_tokens=mb.skip_special_tokens)
            for batch in generated
        )

        neutral_comments = itertools.chain.from_iterable(
            mb.postprocessor(x) if mb.postprocessor else x for x in generated_decoded
        )

        results = scorer.score(test_data["comment"], neutral_comments, verbose=verbose)

        model_results.append(
            {**{"generate_params": params, "model": model_name}, **results}
        )

        write_results(model_results, results_file)

    return model_results


def make_eval_results(result: dict) -> list[EvalResult]:
    eval_results = []
    metrics = [
        {**flatdict.FlatDict({k: result[k]}), **{"metric_type": k}} for k in METRIC_KEYS
    ]
    config = flatdict.FlatDict({k: result[k] for k in result.keys() & set(CONFIG_KEYS)})
    for metric in metrics:
        metric_type = metric.pop("metric_type")
        for metric_key, metric_value in metric.items():
            er = EvalResult(
                metric_config=", ".join(
                    [f"{key}={val}" for key, val in config.items()]
                ),
                metric_type=metric_type,
                metric_name=metric_key,
                metric_value=metric_value,
                task_type="defusing_sarcasm",
                dataset_type="text",
                source_name="kaggle",
                source_url="https://www.kaggle.com/datasets/danofer/sarcasm/data",
                dataset_name="custom",
            )
            eval_results.append(er)
    return eval_results


def push_results(model_name: str, model_results: list[dict]):
    logger.info(f"pushing {len(model_results)} results to repo {model_name}")
    eval_results = list(itertools.chain(*[make_eval_results(r) for r in model_results]))

    metadata_update(
        repo_id=model_name,
        metadata={"eval_results": eval_results, "model_name": model_name},
        overwrite=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    login()

    parser = argparse.ArgumentParser()

    parser.add_argument("model")
    parser.add_argument("test_file")
    parser.add_argument(
        "--compute-baseline",
        "-cb",
        action="store_true",
        help="only compute baseline, do not push to hub",
    )
    parser.add_argument("--generate-params", default="generate_params.csv")
    parser.add_argument("--limit", "-l", type=int, help="limit on test examples")
    parser.add_argument("--results-file", "-f", default="scores.csv")
    parser.add_argument("--push-results", "-pr", action="store_true")
    parser.add_argument(
        "--similarity-model",
        "-s",
        default=SarcasmScorer.DEFAULT_SIMILARITY_MODEL,
        help=f"default: {SarcasmScorer.DEFAULT_SIMILARITY_MODEL}",
    )
    parser.add_argument(
        "--sarcasm-model",
        "-r",
        default=SarcasmScorer.DEFAULT_SARCASM_MODEL,
        help=f"default: {SarcasmScorer.DEFAULT_SARCASM_MODEL}",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="print input and output"
    )

    parser.add_argument("--sarcasm-label", "-b", default="LABEL_1")
    args = parser.parse_args()

    if args.push_results and args.compute_baseline:
        raise ValueError(
            "you called with --compute-baseline and --push-results but the baseline won't be pushed to the hub"
        )

    test_file_name = args.test_file
    limit = args.limit

    if os.path.exists(args.results_file):
        logger.info(f"results file {args.results_file} exists already, appending")

    logger.info(f"detected device: {DEVICE}")
    logger.info(f"loading file: {test_file_name}")
    logger.info(f"considering only first {limit} entries")

    test_data = pd.read_csv(test_file_name, nrows=limit)

    test_data = test_data[["comment", "neutral_comment"]].dropna()

    scorer = SarcasmScorer(
        similarity_model_name=args.similarity_model,
        sarcasm_model_name=args.sarcasm_model,
        sarcasm_label=args.sarcasm_label,
    )

    generate_params = []
    if not args.compute_baseline:
        generate_params = pd.read_csv(args.generate_params)

        logger.info("testing with the following generation params:")
        logger.info(f"\n\n{str(generate_params)}")
        generate_params = generate_params.T.to_dict().values()

    scoring_results = score_model(
        args.model,
        scorer,
        test_data,
        generate_params,
        compute_baseline=args.compute_baseline,
        results_file=args.results_file,
        verbose=args.verbose,
    )

    if args.push_results:
        push_results(args.model, scoring_results)
