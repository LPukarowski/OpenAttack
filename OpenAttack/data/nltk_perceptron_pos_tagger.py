"""
:type: function
:Size: 2.41MB

Model files for pos tagger in nltk.
`[code] <https://github.com/sloria/textblob-aptagger>`__
"""
from OpenAttack.utils import make_zip_downloader
import os

NAME = "TProcess.NLTKPerceptronPosTagger"

URL = "/TAADToolbox/averaged_perceptron_tagger.pickle.zip"
DOWNLOAD = make_zip_downloader(URL, "averaged_perceptron_tagger.pickle")


def LOAD(path):
    import nltk
    import pickle
    ret = nltk.tag.PerceptronTagger(load=False)
    with open(os.path.join(path, "averaged_perceptron_tagger.pickle"), "rb") as f:
        weights = pickle.load(f)
    ret.model.weights = weights[0]
    ret.tagdict = weights[1]
    ret.classes = weights[2]
    return ret.tag
