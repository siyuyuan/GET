# GET

Code for our EMNLP 2022 paper: Generative Entity Typing with Curriculum Learning.

## Dependencies

- torch >= 1.7.1
- tranformers >= 4.5.1

## Dataset
sample dataset is release on dataset.rar

## Running GET
```
python main.py
```

## Reference
- transformer: <https://github.com/huggingface/transformers>

## Citation

If you find our paper or resources useful, please kindly cite our paper. If you have any questions, please [contact us](mailto:jjchen19@fudan.edu.cn)!

```latex
@inproceedings{yuan-etal-2022-generative-entity,
    title = "Generative Entity Typing with Curriculum Learning",
    author = "Yuan, Siyu  and
      Yang, Deqing  and
      Liang, Jiaqing  and
      Li, Zhixu  and
      Liu, Jinxi  and
      Huang, Jingyue  and
      Xiao, Yanghua",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.199",
    doi = "10.18653/v1/2022.emnlp-main.199",
    pages = "3061--3073",
    abstract = "Entity typing aims to assign types to the entity mentions in given texts. The traditional classification-based entity typing paradigm has two unignorable drawbacks: 1) it fails to assign an entity to the types beyond the predefined type set, and 2) it can hardly handle few-shot and zero-shot situations where many long-tail types only have few or even no training instances. To overcome these drawbacks, we propose a novel generative entity typing (GET) paradigm: given a text with an entity mention, the multiple types for the role that the entity plays in the text are generated with a pre-trained language model (PLM). However, PLMs tend to generate coarse-grained types after fine-tuning upon the entity typing dataset. In addition, only the heterogeneous training data consisting of a small portion of human-annotated data and a large portion of auto-generated but low-quality data are provided for model training. To tackle these problems, we employ curriculum learning (CL) to train our GET model on heterogeneous data, where the curriculum could be self-adjusted with the self-paced learning according to its comprehension of the type granularity and data heterogeneity. Our extensive experiments upon the datasets of different languages and downstream tasks justify the superiority of our GET model over the state-of-the-art entity typing models. The code has been released on https://github.com/siyuyuan/GET.",
}
```
