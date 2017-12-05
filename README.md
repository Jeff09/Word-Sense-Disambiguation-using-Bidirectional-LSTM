# Word-Sense-Disambiguation-using-Bidirectional-LSTM

File description:

  data.py   preprocessing Senseval2 and Senseval3 dataset, get the input for model4.py, including the sense embedding of the target sense, and forward data and backward arount the central word.
  
  google_data.py   preprocessing Google research dataset - Word Sense disambiguation corpora, get the input for model4.py, including the sense embedding of the target sense, and forward data and backward arount the central word.
  
  model4.py   build bidrection LSTM for word sense disambiguation, using data.py or google_data.py as input, it will output the model. 
  
  globe.py   load the pre-trained Glove word embedding vector for our own dataset
  
  sense_embedding.py   the 100-dimension sense vector of Google - Word Sense Disambiguation corpora
  
  senseval_sense_embedding.py   the 100-dimensiion sense vector of Senseval2 dataset
  
  Final_report.docx   descript the whole idea of this project
  
  data   including Senseval2 and Senseval3 dataset
