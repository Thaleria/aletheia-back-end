# Aletheia Back End

This project contains the back end of **Aletheia** / **Emma**, an automated fact-checking tool that uses artificial intelligence (AI) to significantly shorten the time between the spread of misinformation and its rebuttal. Emma does not act as a new source of facts, but as an instrument that unlocks existing, verified information sources.

![Aletheia logo](https://raw.githubusercontent.com/Thaleria/aletheia-back-end/main/aletheia.jpg "Aletheia")

## Context

In public and political debate, statements are regularly made that are incorrect or deviate from earlier positions, the voting behavior of political parties, or the scientific consensus as documented, for example, by CBS and CPB. This can lead to confusion and disrupt the debate.

## What Aletheia Does

Emma offers a solution by quickly and effectively identifying misinformation and providing context, so that societal dialogue remains based on facts and reliable information. Initially, Emma analyzes debates in the Dutch House of Representatives (Tweede Kamer). Over time, this will be expanded to the broader public and political debate.

By making this data available in real time and presenting it where it is most needed, Emma provides fast and reliable support in debates and decision-making processes. The ultimate goal is to better equip society against the harmful effects of misinformation and contribute to a fact-based public and political dialogue.

## Project Structure

```text
.
|-- README.md
|-- pyproject.toml
|-- poetry.lock
|-- noxfile.py
|-- src/
|   `-- aletheia_back_end/
|       |-- api/
|       |-- middelware/
|       |-- modules/
|       |-- utils/
|       |-- app.py
|       |-- app_settings.py
|       `-- global_settings.py
|-- tests/
`-- logs/
```

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.
