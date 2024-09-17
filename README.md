# Tibert

`Tibert` is a transformers-compatible reproduction from the paper [End-to-end Neural Coreference Resolution](https://aclanthology.org/D17-1018/) with several modifications. Among these:

- Usage of BERT (or any BERT variant) as an encoder as in [BERT for Coreference Resolution: Baselines and Analysis](https://aclanthology.org/D19-1588/)
- batch size can be greater than 1
- Support of singletons as in [Adapted End-to-End Coreference Resolution System for Anaphoric Identities in Dialogues](https://aclanthology.org/2021.codi-sharedtask.6)
- Hierarchical merging as in [Coreference in Long Documents using Hierarchical Entity Merging](https://aclanthology.org/2024.latechclfl-1.2/)

  
It can be installed with `pip install tibert`.


# Documentation

## Simple Prediction Example

Here is an example of using the simple prediction interface:

```python
from tibert import BertForCoreferenceResolution, predict_coref_simple
from tibert.utils import pprint_coreference_document
from transformers import BertTokenizerFast

model = BertForCoreferenceResolution.from_pretrained(
    "compnet-renard/bert-base-cased-literary-coref"
)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

annotated_doc = predict_coref_simple(
    "Sli did not want the earpods. He didn't like them.", model, tokenizer
)

pprint_coreference_document(annotated_doc)
```

results in:

`>>> (0 Sli ) did not want the earpods. (0 He ) didn't like them.`


## Batched Predictions for Performance

A more advanced prediction interface is available:

```python
from transformers import BertTokenizerFast
from tibert import predict_coref, BertForCoreferenceResolution

model = BertForCoreferenceResolution.from_pretrained(
    "compnet-renard/bert-base-cased-literary-coref"
)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

documents = [
    "Sli did not want the earpods. He didn't like them.",
    "Princess Liana felt sad, because Zarth Arn was gone. The princess went to sleep.",
]

annotated_docs = predict_coref(documents, model, tokenizer, batch_size=2)

for doc in annotated_docs:
    pprint_coreference_document(doc)
```

results in:

`>>> (0 Sli ) did not want the earpods . (0 He ) didn't like them .`

`>>> (0 Princess Liana ) felt sad , because (1 Zarth Arn ) was gone . (0 The princess) went to sleep .`


## Using Coreference Chains

The coreference chains predicted can be accessed using the `.coref_chains` attribute:

```python
annotated_doc = predict_coref_simple(
    "Princess Liana felt sad, because Zarth Arn was gone. The princess went to sleep.",
    model,
    tokenizer
)
print(annotated_doc.coref_chains)
```

`>>>[[Mention(tokens=['The', 'princess'], start_idx=11, end_idx=13), Mention(tokens=['Princess', 'Liana'], start_idx=0, end_idx=2)], [Mention(tokens=['Zarth', 'Arn'], start_idx=6, end_idx=8)]]`


## Hierarchical Merging

Hierarchical merging allows to reduce RAM usage and computations when performing inference on long documents. To do so, the user provides the text cut in chunks. The model will perform prediction for chunks, which means the long document wont be taken at once into memory. Then, hierarchical merging will try to merge chunk predictions. This allow scaling to arbitrarily large documents. See [Coreference in Long Documents using Hierarchical Entity Merging](https://aclanthology.org/2024.latechclfl-1.2/) for more details.

Hierarchical merging can be used as follows:

```python
from tibert import BertForCoreferenceResolution, predict_coref
from tibert.utils import pprint_coreference_document
from transformers import BertTokenizerFast

model = BertForCoreferenceResolution.from_pretrained(
    "compnet-renard/bert-base-cased-literary-coref"
)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

chunk1 = "Princess Liana felt sad, because Zarth Arn was gone."
chunk2 = "She went to sleep."

annotated_doc = predict_coref(
    [chunk1, chunk2], model, tokenizer, hierarchical_merging=True
)

pprint_coreference_document(annotated_doc)
```

This results in:

`>>>(1 Princess Liana ) felt sad , because (0 Zarth Arn ) was gone . (1 She ) went to sleep .`

Even if the mentions `Princess Liana` and `She` are not in the same chunk, hierarchical merging still resolves this case correctly.


## Training a model

Aside from the `tibert.train.train_coref_model` function, it is possible to train a model from the command line. Training a model requires installing the `sacred` library. Here is the most basic example:

```sh
python -m tibert.run_train with\
       dataset_path=/path/to/litbank/repository\
       out_model_dir=/path/to/output/model/directory
```

The following parameters can be set (taken from `./tibert/run_train.py` config function):

| Parameter                    | Default Value       |
|------------------------------|---------------------|
| `batch_size`                 | `1`                 |
| `epochs_nb`                  | `30`                |
| `dataset_name`               | `"litbank"`         |
| `dataset_path`               | `"~/litbank"`       |
| `mentions_per_tokens`        | `0.4`               |
| `antecedents_nb`             | `350`               |
| `max_span_size`              | `10`                |
| `mention_scorer_hidden_size` | `3000`              |
| `sents_per_documents_train`  | `11`                |
| `mention_loss_coeff`         | `0.1`               |
| `bert_lr`                    | `1e-5`              |
| `task_lr`                    | `2e-4`              |
| `dropout`                    | `0.3`               |
| `segment_size`               | `128`               |
| `encoder`                    | `"bert-base-cased"` |
| `out_model_dir`              | `"~/tibert/model"`  |
| `checkpoint`                 | `None`              |


One can monitor training metrics by adding run observers using command line flags - see `sacred` documentation for more details.


# Method

We reimplemented the model from [Lee et al., 2017](https://aclanthology.org/D17-1018/) from scratch, but used BERT as the encoder as in [Joshi et al., 2019](https://aclanthology.org/D19-1588/). We do not use higher order inference as in [Lee et al., 2018](https://aclanthology.org/N18-2108/) since it was found to be not necessarily useful by [Xu and Choi, 2020](https://aclanthology.org/2020.emnlp-main.686/).

## Singletons

Unfortunately, the framework from [Lee et al., 2017](https://aclanthology.org/D17-1018/) cannot represent singletons. This is because the authors were working on the OntoNotes dataset, where singletons are not annotated. We wanted to work on Litbank, so we had to find a way to represent singletons.

We opted to do as in [Xu and Choi, 2021](https://aclanthology.org/2021.codi-sharedtask.6/): we consider mention with a high enough mention scores as singletons, even when they are in no clusters. To force the model to learn proper mention scores, we add an auxiliary loss on mention score (as in [Xu and Choi, 2021](https://aclanthology.org/2021.codi-sharedtask.6/)). To counter dataset imbalance between positive and negative mentions, we opt to compute a weighted loss instead of performing sampling.

## Additional Features

Several work make use of additional features. For now, only the distance between spans is implemented.


# Results

The following table presents the results we obtained on Litbank by training this model. We evaluate on 10% of Litbank documents, each of which consists of ~2000 tokens. The *split* column indicate whether documents were split in blocks of 512 tokens. The *HM* coumns indicates whether we use hierarchical merging.

| Dataset | Base model        | split | HM  | MUC   | B3    | CEAF  | BLANC | LEA   | time (m:s) |
|---------|-------------------|-------|-----|-------|-------|-------|-------|-------|------------|
| Litbank | `bert-base-cased` | no    | no  | 75.03 | 60.66 | 48.71 | 62.96 | 32.84 | 22:07      |
| Litbank | `bert-base-cased` | yes   | no  | 73.84 | 49.14 | 47.88 | 48.41 | 27.63 | 16:18      |
| Litbank | `bert-base-cased` | yes   | yes | 74.54 | 59.30 | 46.98 | 62.69 | 42.46 | 21:13      |


# Citation

If you use this software in a research project, you can cite Tibert as follows:

```bibtex
@Misc{tibert,
  author = {Amalvy, A. and Labatut, V. and Dufour, R.},
  title = {Tibert},
  year  = {2023},
  url   = {https://github.com/CompNet/Tibert},
}
```
