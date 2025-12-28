
# Data sets for document-level MT

This release contains data sets for experiments with document-level machine translation. The data sets have been used in previous studies and provided here for replicability and comparison with other systems. The data sets are taken from the English-German news translation task at WMT 2019 and the English-German bitext in the OpenSubtitles collection v2016 from OPUS. All data sets are sentence aligned with corresponding lines being aligned to each other. Document boundaries are marked with empty lines (on both sides of the parallel corpus).

The data set has been used in the following publication:

```
@inproceedings{scherrer-tiedemann-loaiciga-2019,
    title = "Analysing concatenation approaches to document-level NMT in two different domains",
    author = {Scherrer, Yves and Tiedemann, J{\"o}rg and Lo{\'a}iciga, Sharid},
    booktitle = "Proceedings of the Third Workshop on Discourse in Machine Translation",
    month = nov,
    year = "2019",
    address = "Hong-Kong",
    publisher = "Association for Computational Linguistics",
}
```

Please, cite that paper if you use the data set in your own work.



## Download

Development and test data are available from github:
https://github.com/Helsinki-NLP/doclevel-MT-benchmark

Training data are packaged and distributed via Zenodo with the object identifier 10.5281/zenodo.3525366
https://zenodo.org/record/3525366


## Training data in train/


* wmt.en-de: English-German training data composed out of Europarl v9, NewsCommentary v14, and Rapid2019 from WMT 2019 + backtranslated data from German news crawls with document boundaries
* wmt.de-en: German-English training data training data composed out of Europarl v9, NewsCommentary v14, and Rapid2019 from WMT 2019 with document boundaries
* ost: English-German bitext taken from OpenSubtitles v2016 from OPUS (disjoint from dev and test)



## Development data in dev/

* wmt: newstest 2015 + newstest 2016
* ost: heldout data from OpenSubtitles
* ost-small: subset of ost (top 5000 lines)



## Test data in test/

* wmt: newstest 2018
* ost: heldout data from OpenSubtitles (disjoint from dev)

The data sets are provided in different formats:

* raw, untokenized text
* tokenized and truecased text (*.tok.??)
* shuffled order with arbitrary document boundaries (*.shuffled.tok.??)
* double-spaced text to make single lines to individual documents (*.nocontext.tok.??)

Shuffling can be undone using the attached unshuffle.py script. 
Note that different versions of Python might result in the wrong way of re-ordering the data.



## Related publications


The subtitle data set has also been used in the following publication:

```
@inproceedings{tiedemann-scherrer-2017-neural,
    title = "Neural Machine Translation with Extended Context",
    author = {Tiedemann, J{\"o}rg  and
      Scherrer, Yves},
    booktitle = "Proceedings of the Third Workshop on Discourse in Machine Translation",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W17-4811",
    doi = "10.18653/v1/W17-4811",
    pages = "82--92",
}
```


## License

This package is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International License.
CC BY-NC-SA 4.0

You should have received a copy of the license along with this
work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

