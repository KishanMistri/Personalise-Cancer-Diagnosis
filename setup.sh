#!/bin/bash

echo "installing unzip...!"
sudo apt-get install zip unzip -y

echo "Installing dependencies"
pip3 install -r requirements.txt

if [ -d "~/nltk_data" ]; then
    echo "Downloading/Refreshing NLTK data..."
    /usr/bin/python3 -c "import nltk;nltk.download('all')"
else
    echo "NTLK data folder already exist..."
fi

echo "Variable Setup"
DATA_DIR=data
DATA_FILE=msk-redefining-cancer-treatment.zip
TRAIN_DATA_DIR=train_data
TEST_DATA_DIR=test_data
PROCESSED_DATA_DIR=preprocessed
MODELS_DIR=models

WEBPORT=8080
MAXUPLOADSIZE=500
MAXMSGSIZE=500
STARTPOINT='Home.py'

if [ -d "$DATA_DIR" ]; then
    echo "$DATA_DIR directory exists."
else
    echo "$DATA_DIR directory does not exist.\n Creating one...!!!"
    mkdir $DATA_DIR
    if [ -f "$DATA_FILE" ]; then
        echo "$DATA_FILE file exists."
    else
        echo "$DATA_FILE file does not exist."
        # Setup kaggle.json for this: Refer - https://www.kaggle.com/general/51898#814678
        # kaggle competitions download -c msk-redefining-cancer-treatment
        
        python3 data_download.py
        unzip zip_data.zip 
        unzip $DATA_FILE -d $DATA_DIR
        echo $(ls -al $DATA_DIR)
        rm -rf $DATA_FILE zip_data.zip
    fi
fi

if [ ! -d $TRAIN_DATA_DIR ]; then
    echo "Setting up train data dir"
    mkdir $TRAIN_DATA_DIR
    echo "Unziping train data...!"
    unzip "$DATA_DIR/training_text.zip" -d $TRAIN_DATA_DIR
    unzip "$DATA_DIR/training_variants.zip" -d $TRAIN_DATA_DIR
    rm -rf "$DATA_DIR/training_text.zip" "$DATA_DIR/training_variants.zip"
fi

if [ ! -d $TEST_DATA_DIR ]; then
    echo "Setting up test data dir"
    mkdir $TEST_DATA_DIR
    echo "Unziping test data...!"
    unzip "$DATA_DIR/test_text.zip" -d $TEST_DATA_DIR
    unzip "$DATA_DIR/test_variants.zip" -d $TEST_DATA_DIR
    rm -rf "$DATA_DIR/test_text.zip" "$DATA_DIR/test_variants.zip"
fi

if [ ! -d $PROCESSED_DATA_DIR ]; then
    mkdir $PROCESSED_DATA_DIR
    unzip preprocessed_zip.zip -d $PROCESSED_DATA_DIR
    rm -rf preprocessed_zip.zip
fi

if [ ! -d $MODELS_DIR ]; then
    mkdir $MODELS_DIR
    unzip models_zip.zip -d $MODELS_DIR
    rm -rf models_zip.zip
fi
echo "Running Application"
nohup streamlit run $STARTPOINT --server.port $WEBPORT --server.maxUploadSize $MAXUPLOADSIZE --server.maxMessageSize $MAXMSGSIZE --theme.base "dark" &
echo "Application live on port $WEBPORT"
