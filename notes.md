To convert the notebook to README.md (excluding code cells:)

```
uvr jupyter nbconvert notebook.ipynb --to markdown --TagRemovePreprocessor.remove_input_tags=all --TemplateExporter.exclude_input=True --output README.md 
```