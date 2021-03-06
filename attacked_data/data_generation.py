!pip install textattack --upgrade
!pip install transformers datasets
!pip install tqdm==4.61

# Import statements from models
import datasets
import textattack
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModel
from textattack.attack_recipes import BAEGarg2019, CheckList2020, DeepWordBugGao2018, HotFlipEbrahimi2017, Kuleshov2017, PSOZang2020, PWWSRen2019, TextFoolerJin2019, TextBuggerLi2018, GeneticAlgorithmAlzantot2018, FasterGeneticAlgorithmJia2019, BERTAttackLi2020, IGAWang2019, InputReductionFeng2018
from textattack.models.wrappers import HuggingFaceModelWrapper, ModelWrapper
from textattack.models.helpers import WordCNNForClassification, LSTMForClassification

#@title Experiment Options
model_chosen = "bert-base-uncased" #@param["bert-base-uncased", "lstm","cnn"]
attack = "PSO" #@param["GeneticAlgo", "FasterGeneticAlgo", "BAE", "BERTAttack", "CheckList", "DeepWordBug", "HotFlip", "IGA", "InputReduction", "Kuleshov2017", "PSO", "PWWS", "TextFooler", "TextBugger"]

# Get the model_wrapper to use with or without the attack
if model_chosen == "bert-base-uncased":
  tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
  model_chosen = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
  model_wrapper = HuggingFaceModelWrapper(model_chosen, tokenizer)
elif model_chosen == "cnn":
  model_chosen = WordCNNForClassification
  tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
  model_wrapper = textattack.models.wrappers.ModelWrapper() #WordCNNForClassification, tokenizer)
elif model_chosen == "lstm":
  model_chosen = LSTMForClassification
  tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
  model_wrapper = textattack.models.wrappers.ModelWrapper() #LSTMForClassification, tokenizer)

# Get the model from the attack recipe chosen 
if attack == "GeneticAlgo":
  model = GeneticAlgorithmAlzantot2018.build(model_wrapper)
elif attack == "FasterGeneticAlgo":
  model = FasterGeneticAlgorithmJia2019.build(model_wrapper)
elif attack == "BAE":
  model = BAEGarg2019.build(model_wrapper)
elif attack == "BERTAttack":
  model = BERTAttackLi2020.build(model_wrapper) 
elif attack == "CheckList":
  model = CheckList2020.build(model_wrapper)
elif attack == "DeepWordBug":
  model = DeepWordBugGao2018.build(model_wrapper)
elif attack == "HotFlip":
  model = HotFlipEbrahimi2017.build(model_wrapper)
elif attack == "IGA":
  model = IGAWang2019.build(model_wrapper)
elif attack == "InputReduction":
  model = InputReductionFeng2018.build(model_wrapper)
elif attack == "Kuleshov2017":
  model = Kuleshov2017.build(model_wrapper)
elif attack == "PSO":
  model = PSOZang2020.build(model_wrapper)
elif attack == "PWWS":
  model = PWWSRen2019.build(model_wrapper)
elif attack == "TextFooler":
  model = TextFoolerJin2019.build(model_wrapper)
elif attack == "TextBugger":
  model = TextBuggerLi2018.build(model_wrapper)

# Load the dataset
dataset = textattack.datasets.HuggingFaceDataset("imdb", split="train")

# Attack the dataset and save to attacked_data
attack_args = textattack.AttackArgs(num_examples=100, shuffle=True, silent=True)
attacker = textattack.Attacker(model, dataset, attack_args=attack_args)
attacker.attack_dataset()
