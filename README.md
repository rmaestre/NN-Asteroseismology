# NN-Asteroseismology
Neural Network application on Asteroseismology.

Simple example to preprocess raw star data (if needed), read tf.Dataset (with all files included) and train a DSWConvolution model

```
from astronn.preprocess import predeltascuti, precorot
from astronn.datasets import deltascuti, starmodels, corot
from astronn.models import separableconvnn

import matplotlib.pyplot as plt


preprocessor = precorot(
        conf_file="astronn/data/corot/parameters.csv", cols=["corot", "loggs"]
    )
preprocessor.preprocess_files(
    input_folder="astronn/data/corot/raw/*",
    output_folder="astronn/data/corot/preprocessed/",
)

if True:
    # Preprocess 77 corot stars
    preprocessor = precorot(
        conf_file="astronn/data/corot/parameters.csv", cols=["corot", "loggs"]
    )
    preprocessor.preprocess_files(
        input_folder="astronn/data/corot/raw/*",
        output_folder="astronn/data/corot/preprocessed/",
    )

if False:
    # Read datasets of preprocessed stars
    cr = corot()
    df_corot = cr.load("astronn/data/deltascuti/preprocessed/*", batch_size=1)
    for line in df_corot.take(1):
        print(line[0].shape)
        print(line[1].shape)
        print(line[0].numpy())


if False:
    # Preprocess eleven delta scuti stars
    preprocessor = predeltascuti()
    preprocessor.preprocess_files(
        input_folder="astronn/data/deltascuti/raw/*",
        output_folder="astronn/data/deltascuti/preprocessed/",
    )

if False:
    # Read datasets of preprocessed stars
    ds = deltascuti()
    df_ds = ds.load("astronn/data/deltascuti/preprocessed/*", batch_size=1)
    for line in df_ds.take(1):
        print(line[0].shape)
        print(line[1].shape)
        print(line[0].numpy())

# Load star models dataset
sm = starmodels()
sm_df = sm.load("/home/roberto/Downloads/dataall_parts/*")

# Split train / test
# train_dataset = sm_df.take(800)
# test_dataset = sm_df.skip(100)

sepconv_mod = separableconvnn()  # init model
sepconv_mod.compile(learning_rate=0.0001)  # compile model

# fit model
history = sepconv_mod.model.fit(sm_df, steps_per_epoch=50, epochs=1000)
```
