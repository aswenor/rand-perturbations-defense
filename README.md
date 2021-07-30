# Using Random Perturbations to Mitigate Adversarial Attacks on NLP Models

## Motivation
Deep learning models have excelled in solving many difficult problems in Natural Language Processing (NLP), but it has been demonstrated such models to be susceptible to extensive vulnerabilities. These NLP models are vulnerable to adversarial attacks which is exacerbated by the use of public datasets that typically are not manually inspected before use. We offer two defense methods against these attacks that use random perturbations. We performed tests that showed our Random Perturbations Defense and our Increased Randomness Defense to both be effective in returning attacked models to their original accuracy (before attacks) for 6 different attacks from the TextAttack library. 

## Dependencies 
#### Defense Methods
* Natural Language Toolkit (nltk)
* pyspellchecker
* transformers
* clean-text

#### Data Generation
* TextAttack
* transformers
* tqdm 4.61

## Data & Data Generation
We use the TextAttack library to generate data to test on. The data we used to test our methods can be found in the attacked_data folder. The following attack methods were used to generate 100 perturbed reviews from the IMDB dataset:
* BERT-Based Adversarial Examples (BAE)
* DeepWordBug
* Faster Alzantot Genetic Algorithm
* Kuleshov
* Probability Weighted Word Saliency (PWWS)
* TextBugger
* TextFooler

More data can be generated using our data_generation file that exists in the attacked_data folder. You must select the `model_wrapper` to use and the attack recipe `model` you would like to use. You can change the number of examples to create by changing the `num_examples` variable in this line:
```python
attack_args = textattack.AttackArgs(num_examples=100, shuffle=True, silent=True)
```
## Using our Defense Methods
Both methods exist within the defense_methods folder. Our Random Peturbations Defense has options for tuning with the parameters `l` and `k` which have the following properties:
* `l` = number of replicates made for each sentence in a review
* `k` = number of random corrections made for each replicate

We recommend using `l` = 7 and `k` = 5 for a starting point as this is what we recieved our best results with. You can also choose which attack method you would like to test on. The perturbed data in the attacked_data folder is preset as options in our existing code. You can change the value of `data_to_use` to choose the data you want to use for your tests.

Our Increased Randomness Defense has options for tuning with a single parameter `k` which has the following property:
* `k` = number of replicates made from randomly selected sentences in a review

We recommend using `k` = 41 for a starting point as this is what we recieved our best results with. Again, you can choose which attack method you would like to test on using the same steps as the previous defense method. 

## Credits
[TextAttack Library](https://github.com/QData/TextAttack)\
🤗[HuggingFace Transformers Library](https://huggingface.co/transformers/)
