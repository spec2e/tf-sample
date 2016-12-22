import tempfile
import io
import urllib.request



#train_file = tempfile.NamedTemporaryFile(mode='w')
train_file = io.FileIO("train.csv", "w")

#test_file = tempfile.NamedTemporaryFile(mode='w')
test_file = io.FileIO("test.csv", "w")

urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)