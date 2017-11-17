## Project: Classify Kaggle San Francisco Crime Description

### Highlights:
  - This is a **multi-class text classification (sentence classification)** problem.
  - The goal of this project is to **classify Kaggle San Francisco Crime Description into 39 classes**.
  - This model was built with **CNN, RNN (LSTM and GRU) and Word Embeddings** on **Tensorflow**.

### Data: [Kaggle San Francisco Crime](https://www.kaggle.com/c/sf-crime/data)
  - Input: **Descript**
  - Output: **Category**
  - Examples:

    Descript   | Category
    -----------|-----------
    GRAND THEFT FROM LOCKED AUTO|LARCENY/THEFT
    POSSESSION OF NARCOTICS PARAPHERNALIA|DRUG/NARCOTIC
    AIDED CASE, MENTAL DISTURBED|NON-CRIMINAL
    AGGRAVATED ASSAULT WITH BODILY FORCE|ASSAULT
    ATTEMPTED ROBBERY ON THE STREET WITH A GUN|ROBBERY
    
### Train:
  - Command: python3 train.py train_data.file train_parameters.json
  - Example: ```python3 train.py ./data/train.csv.zip ./training_config.json```

### Predict:
  - Command: python3 predict.py ./trained_results_dir/ new_data.csv
  - Example: ```python3 predict.py ./trained_results_1478563595/ ./data/small_samples.csv```

### Server:
  - Command: python3 server.py ./trained_results_dir
  - Open ``` http://127.0.0.1:5000/ (Press CTRL+C to quit) ```

### References:
 - [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)  
 - See .staging/notes/REFERENCES.md

## Docker 

### Docker Compose Example
======================

This is Docker Compose example. You can boot a container based on this
repository image with Jupyter port 8888, its password 'foobar' and sharing
files in `notebook` directory with the container by just running

```shellsession
docker-compose up
```

To stop it, please press 'Ctrl-C'.

