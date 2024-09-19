# Benchmarking Distributional Alignment of Large Language Models

### [Paper (arxiv)](TODO) | [OpenReview](TODO)


```bibtex
    @Article{TODO
    }
```


## Getting started
You can start by cloning our repository and following these steps. Each step is also documented in an example job script ```job.sh``` that contains the file execution order with example commands.

1. Datasets:
    1. **OpinionQA**: In `./opinions_qa`, you will find our 100 question subset of the 500 contentious OpinionQA questions [(source)](https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69).
    2. **NYT Books**: In `./opinions_qa`, you will find our preprocessed NYT Books dataset where ```data.json``` maps 

3. Compute human and LM opinion distributions using this [notebook](https://github.com/tatsu-lab/opinions_qa/blob/master/process_results.ipynb). 

4. You can explore human-LM alignment along various axes using the following notebooks: [representativeness](https://github.com/tatsu-lab/opinions_qa/blob/master/representativeness.ipynb), [steerability](https://github.com/tatsu-lab/opinions_qa/blob/master/steerability.ipynb), [consistency](https://github.com/tatsu-lab/opinions_qa/blob/master/consistency.ipynb) and [refusals](https://github.com/tatsu-lab/opinions_qa/blob/master/refusals.ipynb).

5. (Optional) If you would like to query models yourself, you will need to set up the [crfm-helm](https://github.com/stanford-crfm/helm) Python package. 

Then, to obtain model responses, run:
```
helm-run -c src/helm/benchmark/presentation/run_specs_opinions_qa_openai_default.conf --max-eval-instances 500 --suite $SUITE
helm-run -c src/helm/benchmark/presentation/run_specs_opinions_qa_ai21_default.conf --max-eval-instances 500 --suite $SUITE
helm-run -c src/helm/benchmark/presentation/run_specs_opinions_qa_openai_steer.conf --max-eval-instances 50000 --suite $SUITE
helm-run -c src/helm/benchmark/presentation/run_specs_opinions_qa_ai21_steer.conf --max-eval-instances 50000 --suite $SUITE
```

# Maintainers

[Nicole Meister](nicolemeister.github.io)


## Code overview

Here is a brief description of the individual components

#### Biased coin flip experiment 
```biased_categories.py```: Calculates bias and identifies the K=20 most biased categories

#### Datasets
```biased_categories.py```: Calculates bias and identifies the K=20 most biased categories

#### Human Annotations
```biased_categories.py```: Calculates bias and identifies the K=20 most biased categories



#### Data processing
```{Dataset}/data_process.py```: Processes the COCO-Stuff, DeepFashion, AwA, UnRel datasets

```split_80_20.py```: Does a 80-20 split of the COCO-Stuff/AwA training set to create a validation set

#### Biased categories identification
```biased_categories.py```: Calculates bias and identifies the K=20 most biased categories

#### Training
```train.py```: Trains various models (standard, cam, featuresplit, removeclabels, removecimages, splitbiased, weighted, negativepenalty, classbalancing, attribdecorr)

#### Evaluation






```evaluate.py```: Evaluates a trained model on the COCO-Stuff, DeepFashion, AwA datasets, on their exclusive and co-occur test distributions

```evaluate_unrel.py```: Evaluates a trained model on the UnRel dataset

```weight_similarity.py```: Calculates the cosine similarity between W_o and W_s to verify that they capture distinct information

```get_cams.py```: Saves class activation maps (CAMs) to understand what the model is looking at
- **Image IDs for Figure 3**: Skateboard (317040), Microwave (191632)
- **Image IDs for Figure A4**: Handbag (167235, 37124), Snowboard (423602, 581921), Car (574087, 119802), Spoon (227858, 42526), Remote (390829, 267116)

```get_prediction_examples.py```: Finds successful and unsuccessful image examples of a model's prediction for a category b
- **Image IDs for Figure A3**: Skateboard (292789, 430096), Microwave (105547, 444275, 110027, 292905), Snowboard (50482, 174103, 526133, 304817)
