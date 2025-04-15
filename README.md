Update: in https://arxiv.org/html/2504.09596v1, I listed the ideas worth to try but not yet due to my limited bandwidth in sparse time.

Pls feel free to do these experiments to have fun, and pls consider citing the article if it somehow helps in your recsys exploration:

```
@article{huang2025revisiting_sasrec,
  title={Revisiting Self-Attentive Sequential Recommendation},
  author={Huang, Zan},
  journal={CoRR},
  volume={abs/2504.09596},
  url={https://arxiv.org/abs/2504.09596},
  eprinttype={arXiv},
  eprint={2504.09596},
  year={2025}
}
```

For questions or collaborations, pls create a new issue in this repo or drop me an email using the email address as shared.

---

update: as expected, with few lines of xavier initialization code added, it converges as fast as original tf version now, pls check github issue of this repo and  https://github.com/pmixer/SASRec.pytorch for more details if interested :)

---

update: a pretrained model added, pls run the command as below to test its performance:

```
python main.py --dataset=ml-1m --train_dir=default --dropout_rate=0.2 --device=cuda --state_dict_path='ml-1m_default/TiSASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --maxlen=200
```

---

modified based on [paper author's tensorflow implementation](https://github.com/JiachengLi1995/TiSASRec), switching to PyTorch(v1.6) for simplicity, executable by:

```python main.py --dataset=ml-1m --train_dir=default --device=cuda```

pls check paper author's [repo](https://github.com/JiachengLi1995/TiSASRec) for detailed intro and more complete README, and here's paper bib FYI :)

```
@inproceedings{li2020time,
  title={Time Interval Aware Self-Attention for Sequential Recommendation},
  author={Li, Jiacheng and Wang, Yujie and McAuley, Julian},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={322--330},
  year={2020}
}
```
