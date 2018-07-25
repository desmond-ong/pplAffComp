import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_IMG_DIR = os.path.join(UTILS_DIR, "..", "..", "CognitionData", "faces")
FACE_OUTCOME_EMOTION_PATH = os.path.join(UTILS_DIR, "..", "..",
                                         "CognitionData", "data_faceWheel.csv")
WORD_OUTCOME_EMOTION_PATH = os.path.join(UTILS_DIR, "..", "..",
                                         "CognitionData",
                                         "dataSecondExpt_utteranceWheel.csv")

OUTCOME_VAR_NAMES = ['payoff1', 'payoff2', 'payoff3', 
                     'prob1', 'prob2', 'prob3', 
                     'win', 'winProb', 'angleProp']
EMOTION_VAR_NAMES = ['happy', 'sad', 'anger', 'surprise', 
                     'disgust', 'fear', 'content', 'disapp']

OUTCOME_VAR_DIM = len(OUTCOME_VAR_NAMES)
EMOTION_VAR_DIM = len(EMOTION_VAR_NAMES)

class MultimodalDataset(Dataset):
    """A multimodal experimental dataset."""
    
    def __init__(self, csv_file, embeddings=None, img_dir=None,
                 transform=None):
        """
        Args:
            csv_file (string): Path to the experiment csv file 
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.expdata = pd.read_csv(csv_file)
        self.embeddings = embeddings
        self.img_dir = img_dir
        self.transform = transform
        
        self.has_utterances = False
        self.has_faces = False
        self.has_emotions = False
        self.has_outcomes = False
        
        # Check if dataset has utterances
        if "utterance" in self.expdata.columns and embeddings is not None:
            self.has_utterances = True
        # Check if dataset has face images
        if "facePath" in self.expdata.columns and img_dir is not None:
            self.has_faces = True
        # Check if dataset has emotion ratings
        if set(EMOTION_VAR_NAMES).issubset(self.expdata.columns):
            self.has_emotions = True
            self.normalize_emotions()
        # Check if dataset has outcomes
        if set(OUTCOME_VAR_NAMES).issubset(self.expdata.columns):
            self.has_outcomes = True
            self.normalize_outcomes()

    def __len__(self):
        return len(self.expdata)

    def __getitem__(self, idx):

        if self.has_emotions:
            emotions = np.array(self.expdata.iloc[idx]["happy":"disapp"],
                                np.float32)
        else:
            emotions = 0

        if self.has_outcomes:
            outcomes = np.array(self.expdata.iloc[idx]["payoff1":"angleProp"],
                                np.float32)
        else:
            outcomes = 0

        if self.has_utterances:
            word = self.expdata.iloc[idx]["utterance"]
            embed = self.embeddings[word]
        else:
            word = ""
            embed = 0
        
        if self.has_faces:
            img_name = os.path.join(self.img_dir,
                                    self.expdata.iloc[idx]["facePath"]
                                    + ".png")
            try:
                image = Image.open(img_name).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            except:
                print(img_name)
                raise
        else:
            image = 0
            
        return word, embed, image, emotions, outcomes
    
    def normalize_outcomes(self):
        """Normalizes outcome data.
        
        payoff1, payoff2, payoff3 and win are between 0 and 100
        need to normalize to [0,1] to match the rest of the variables,
        by dividing payoff1, payoff2, payoff3 and win by 100.
        """        
        self.expdata.loc[:,"payoff1"] = self.expdata.loc[:,"payoff1"]/100
        self.expdata.loc[:,"payoff2"] = self.expdata.loc[:,"payoff2"]/100
        self.expdata.loc[:,"payoff3"] = self.expdata.loc[:,"payoff3"]/100
        self.expdata.loc[:,"win"]     = self.expdata.loc[:,"win"]/100
    
    def normalize_emotions(self):
        """Normalize emotion ratings.
        
        Emotions were rated on a 1-9 Likert scale.
        use emo <- (emo-1)/8 to transform to within [0,1]
        """
        self.expdata.loc[:,"happy":"disapp"] = \
            (self.expdata.loc[:,"happy":"disapp"]-1)/8


def load_face_outcome_emotion_data(batch_size,
                                   csv_file=FACE_OUTCOME_EMOTION_PATH,
                                   img_dir=FACE_IMG_DIR):
    """Loads the face/outcome/emotion dataset."""
        
    # Note that we downsample to 64 x 64 here
    # Because we wanted a nice power of 2 
    # (and DCGAN architecture assumes input image of 64x64) 
    img_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor()
        ])

    # Reads in datafiles
    dataset = MultimodalDataset(csv_file=csv_file, img_dir=img_dir,
                                transform=img_transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

    return dataset, loader

def load_word_outcome_emotion_data(batch_size, embeddings,
                                   csv_file=WORD_OUTCOME_EMOTION_PATH):
    # Read in datafiles
    dataset = MultimodalDataset(csv_file=csv_file, embeddings=embeddings)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

    return dataset, loader
