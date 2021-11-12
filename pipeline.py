from typing import List, Optional, Tuple, Union
import pandas as pd
from spacy import load as spacy_load
from tqdm.auto import tqdm
from torch.optim import AdamW

from FeatureExtractor import FeatureExtractor
from FeatureExtractor.grammar_rules import preload_lexique3_freq_table
from ml.dataset import FeatureDataset
from ml.models.featureClassifier import FeatureClassifier
from ml.logger import Logger
from ml.callbacks import Callbacks


DataFrame = pd.DataFrame


def data_to_dataframe_with_features(
    questions : List[str],
    labels : List[int],
    restricted : bool = True,
) -> DataFrame:
    """
    Turns a list of (questions) and their (labels) into a dataframe with lexical-based features extracted from the questions.
    The quantity of features extracted depends on the (restricted) flag.
    For non-literary or short texts, it is recommended to set (restricted) as True.
    """
    # Get and arrange the features names
    features = list(FeatureExtractor('Ceci est une phrase de test.', restricted=restricted).extract_features(avoid=['_feature_8']).loc[:, 'name'])
    for i, feature in enumerate(features):
        if feature.startswith('Moyenne de la fréquence des ') or feature.startswith('Ecart type de la fréquence des '):
            feature = feature.replace('Moyenne de la fréquence des ', 'M_')
            feature = feature.replace('Ecart type de la fréquence des ', 'SD_')
            feature = feature.replace(' dans le texte selon L3ST', '')
        features[i] = feature
    # DF initialization
    dataframe = pd.DataFrame({'question' : questions, 'label' : labels})
    dataframe.loc[:, features] = 0
    # Loading feature extraction resources
    nlp = spacy_load('fr_core_news_md')
    frequency_table = preload_lexique3_freq_table()
    # Loop for filling in the features
    for i, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        dataframe.loc[i, features] = list(FeatureExtractor(row['question'], nlp, frequency_table, restricted).extract_features(avoid=['_feature_8'], verbose=False).loc[:, 'value'])
    return dataframe


def dataframe_to_dataset(
    dataframe_with_features : DataFrame,
    question_column : str = 'question',
    label_column : str = 'label',
    normalize : bool = True,
    na_replace_value : Optional[float] = -1.0,
) -> FeatureDataset:
    """
    Converts a (dataframe_with_features) to a FeatureDataset for fitting a model.
    The columns containing questions and labels are identified with (question_column) and (label_column).
    The dataset is normalized if the (normalize) flag is set to True, ignoring nans. This is strongly recommended.
    The nans are replaced with (na_replace_value) to avoid errors during training, -1.0 by default.
    """
    # Relevant lists
    entries, labels = [], dataframe_with_features.loc[:, label_column]
    # Drop features columns
    dataframe_with_features = dataframe_with_features.drop(columns = [question_column, label_column])
    # Get feature names
    features_names = list(dataframe_with_features.columns)
    # Fill entries
    for _, row in dataframe_with_features.iterrows():
        entries.append(list(row[features_names]))
    # Create the dataset
    dataset = FeatureDataset(features_names, entries, labels)
    # Normalize it
    if normalize:
        dataset.normalize(ignore_na=True)
    # Replace nans
    if na_replace_value is not None:
        dataset.replace_na(na_replace_value)
    return dataset


def split_dataset(
    dataset : FeatureDataset,
    p : float = 0.6,
    balance_train : bool = False,
    balance_val : bool = True,
    seed : Optional[int] = 42,
) -> Tuple[FeatureDataset, FeatureDataset]:
    """
    Splits the (dataset) according to the proportion (p). Training size is int(p * len(dataset)). By default, p = 0.6.
    The training set is balanced if (balance_train) is set to True. Same goes for validation set and (balance_val).
        By default, training isn't balanced because weighting the loss function is enough, and validation is balanced to get relevant metrics.
    For reproducibilty's sake, a (seed) can be passed, or left to None for real randomness.
    """
    # Split the dataset
    train, val = dataset.split(p, seed)
    # Maybe balance the validation set
    if balance_val:
        dropped = val.balance(seed)
        train.add(*dropped)
    # Maybe balance train
    if balance_train:
        train.balance(seed)
    return train, val


def make_model(
    training_dataset : FeatureDataset,
    dropout_rate : float = 0.4,
    hidden_size : int = 200,
    seed : Optional[int] = 42,
    use_weighted_loss : bool = True,
    cuda : bool = True,
) -> FeatureClassifier:
    """
    Generates a FeatureClassifier model with the given parameters.
    The (training_dataset) is needed for two reasons: knowing the number of features and weighting the loss according to the dataset proportions.
    If you want to disable loss weighting, set (use_weighted_loss) to False.
    The hyperparameters are the (dropout_rate) and the (hidden_size), both set to the best value found by cross-training.
    For reproducibilty's sake, a (seed) can be passed, or left to None for real randomness.
    The model is switched to cuda unless (cuda) is set to False.
    """
    number_of_features=training_dataset.number_of_features()
    weights = 1 - training_dataset.proportions() if use_weighted_loss else None
    model = FeatureClassifier(
        number_of_features=number_of_features, 
        dropout=dropout_rate, 
        hidden_size=hidden_size,
        weights=weights,
        seed=seed,
    )    
    if cuda:
        model.cuda()
    return model


def make_fitting_related_objects(
    model : FeatureClassifier,
    learning_rate : float = 4e-5,
    metrics : Optional[List[str]] = None,
    early_stopping_patience : int = 500,
    early_stopping_warmup : int = 100,
    early_stopping_metric : str = 'Validation Accuracy',
    restore_best_weights : bool = True,
    best_weights_savepath : str = '',
) -> Tuple[AdamW, Logger, Callbacks]:
    """
    Generates the objects needed for the model fitting, namely an AdamW optimizer, a Logger and Callbacks.
    The (model) is needed for the optimizer to know what to optimize.
    The optimizer is an AdamW with (learning_rate) set to 4e-5 by default, value found by cross-training.
    The (metrics) list tells the Logger what to monitor, by default training and validation loss and accuracy.
    The (early_stopping_patience) can be left to 0 if you don't want early stopping. If you do, set it to an integer >= 1. 
    If early stopping is on, you can also choose:
        - the number of warmup epochs with (early_stopping_warmup).
        - the metric used with (early_stopping_metric), the metric used needs to be in the logger, so pass it with (metrics).
        - whether or not to (restore_best_weight) found during training in regards to (early_stopping_metric)
        - if (restore_best_weight) is on, please provide a (best_weights_savepath) in which the model and optimizer will be saved
    """
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # Logger
    if metrics is None:
        metrics = ['train loss', 'train acc', 'val loss', 'val acc']
    logger = Logger(metrics)
    # Callbacks
    callbacks = Callbacks(best_weights_savepath)
    if early_stopping_patience > 0:
        callbacks.add_early_stopping(
            patience=early_stopping_patience,
            metric_name=early_stopping_metric,
            warmup=early_stopping_warmup,
            restore_best_weights=restore_best_weights,
        )
    return optimizer, logger, callbacks


def fit_the_model(
    model : FeatureClassifier,
    fitting_related_objects : Tuple[AdamW, Logger, Callbacks],
    training_and_validation_datasets : Tuple[FeatureDataset, FeatureDataset],
    number_of_epochs : int = 10000,
    batch_size : int = 256,
    seed : Optional[int] = 42,
) -> None:
    """
    Fits the (model) on the (training_and_validation_datasets) with the (fitting_related_objects).
    The (training_and_validation_datasets) is a tuple of two FeatureDataset, like the one returned by split_dataset.
    The (fitting_related_objects) is a tuple of an AdamW, a logger and callbacks, like the one returned by make_fitting_related_objects.
    The fittings lasts for (number_of_epochs) with a given (batch_size), by default 256, best value found with cross training.
    For reproducibilty's sake, a (seed) can be passed, or left to None for real randomness.
    """
    # Unpacking
    optimizer, logger, callbacks = fitting_related_objects
    training_dataset, validation_dataset = training_and_validation_datasets
    model.fit(
        optimizer=optimizer,
        logger=logger,
        epochs=number_of_epochs,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        batch_size=batch_size,
        callbacks=callbacks,
        seed=seed,
    )
    # Display
    for metric_name, metric in logger.metrics.items():
        print(f"Best {metric_name}: {metric.best(metric.history)}")
    print()
    logger.plot('loss')
    logger.plot('accuracy')


def pipeline_train(
    questions : List[str],
    labels : List[int],
    p : float = 0.6,
    seed : Optional[int] = 42,
    best_weights_savepath : str = 'temp_best_weight.pt',
    savepath : Optional[str] = None,
) -> FeatureClassifier:
    """
    Standardized pipeline for training a FeatureClassifier on a list of (questions) associated to the given (labels).
    The arguments for all the pipeline steps are the default ones, but some are included because of their importance:
        - (p), the training proportion,
        - (seed), for reproducibility's sake,
        - (best_weight_savepath), where we temporary save the best weights during training,
    If you want to save the model we get by the end of the pipeline, pass a (savepath) argument. It will be valid for pipeline_predict.
    """
    dataframe = data_to_dataframe_with_features(questions=questions, labels=labels)
    dataset = dataframe_to_dataset(dataframe_with_features=dataframe)
    train, val = split_dataset(dataset=dataset, p=p, seed=seed)
    model = make_model(training_dataset=train, seed=seed)
    fitting_related_objects = make_fitting_related_objects(model, best_weights_savepath=best_weights_savepath)
    fit_the_model(
        model=model, 
        fitting_related_objects=fitting_related_objects,
        training_and_validation_datasets=(train, val),
        seed=seed,
    )
    # Eventual save
    if savepath is not None:
        model.save(savepath)
    return model


def pipeline_predict(
    questions : List[str],
    model_or_path : Union[str, FeatureClassifier],
    batch_size : int = 64,
) -> List[int]:
    """
    Standardized pipeline for prediction of the quality of a list of (questions).
    The model used for prediction can be passed directly or pointed to, both with the (model_or_path) argument.
    The (batch_size) for predictions shouldn't be a problem, as the model is fast and lightweight.
    """
    # If the model_or_path is a path, loads the model
    model = FeatureClassifier.load(model_or_path) if isinstance(model_or_path, str) else model_or_path
    # Format the data as a dataset
    dataset = dataframe_to_dataset(data_to_dataframe_with_features(questions, [0 for _ in questions]))
    # Return the predictions
    return model.predict(dataset, batch_size)