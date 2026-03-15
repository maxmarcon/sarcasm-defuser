#! /usr/bin/env bash

# creates the README.md file from the content of the jupyter notebook

uv run jupyter nbconvert notebook.ipynb \
--to markdown \
--TagRemovePreprocessor.remove_input_tags=all \
--TemplateExporter.exclude_input=True \
--output README_temp.md \
--NbConvertApp.output_files_dir media \
&& cat header.md README_temp.md > README.md && rm README_temp.md