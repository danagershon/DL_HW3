r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(  
        batch_size=512,  # largeer batch size => speed training
        seq_len=64,
        h_dim=128,  # smaller h_dim => speed training
        n_layers=2,  # fewer layers => speed training
        dropout=0.3,  
        learn_rate=0.002,
        lr_sched_factor=0.1,
        lr_sched_patience=3,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = 'ACT I.'
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training on the whole text for a few reasons:

1. training on the whole text can be infeasible to store in memory for large datasets. 
Breaking the text into smaller sequences makes it managable to store in memory.

2. splitting to smaller sequences allows batch processing, where we process multiple sequences in parallel. 
This speeds up training by having more frequent gradient updates which leads to faster model convergence.

3. training on smaller sequences allows the model to capture local text context.

4. splitting to smaller sequences mitigates the vanishing and exploding gradients problems.

"""

part1_q2 = r"""
**Your answer:**

The generated text shows memory longer than the sequence length thanks to the hidden states,
that carry information across sequences. This allows the model to remember information from previous sequences,
thus it incorporates into the generated text context beyond the current sequence length.

"""

part1_q3 = r"""
**Your answer:**

We do not shuffle the order of batches when training because we deal with sequential data, and we want to preserve 
the temporal dependencies and contextual information it has.
Shuffling can disrupt the inherent sequential structure, thus preventing the model from learning meaningful patterns.

"""

part1_q4 = r"""
**Your answer:**

1. We lower the temperature for sampling to make the generated text less random.
By lowering the temperature, we change the probability distribution of the next character such that
the most likely characters (higher scores) are more probable. By that we reduce the chances of unlikely characters 
to be chosen, making the genereted text more coherent.

2. When the temperature is very high, the probability distribution of the next character becomes more uniform.
This means characters have almost equal probability of being chosen next, regardless of their scores.
The generated text becomes more randomized and less coherent, but a benefit is that the text can be more diverse.

3. When the temperature is very low, the probability distribution of the next character becomes skewed towards the 
most likely characters (higher scores), making them almost certain to be chosen. This increases the predictability of
the generated text but reduces variance, making the model prone to repeat patterns.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, 
        z_dim=0, 
        discriminator_optimizer=0, 
        generator_optimizer=0, 
        data_label=0,
        label_noise=0, 
    )

    hypers = dict(
        batch_size=32, 
        z_dim=128, 
        discriminator_optimizer={
            'type': 'Adam',
            'lr': 0.0001,
            'betas': (0.5, 0.999)
        }, 
        generator_optimizer={
            'type': 'Adam',
            'lr': 0.0001,
            'betas': (0.5, 0.999)
        }, 
        data_label=1.0,  # Assuming real data is labeled as 1
        label_noise=0.1  # Small noise to the labels
    )

    return hypers


part2_q1 = r"""
**Your answer:**

1. We maintain gradients when sampling from the GAN during the generator update.
This is because in order for the generator to learn and improve, we need to update the generator's parameters, for which
we need to compute the gradients of the generator's loss w.r.t to its parameters.

2. We discard gradients when sampling from the GAN during the discriminator update.
This is because we do not want to update the generator's parameters, thus we do not need to maintain the gradients 
for the generator.

"""

part2_q2 = r"""
**Your answer:**

1. We should not decide to stop training solely based on the fact that the Generator loss is below some threshold.
This is because low generator loss only indicates that the generator is successful at fooling the discriminator, and
it does not mean that the images the generator generates are realistic. 
Another point to consider is that the generator loss can be low while the discriminator is high, indicating that 
the discriminator is weak, and the generator is not being effectively challenged.

2. If the discriminator loss remains at a constant value while the generator loss decreases, it theoretically 
means that the discriminator is not improving while the generator is improving, but the underlying case might be different.
If the generator outputs a limited variety of images that it found fools the discriminator, then the generator loss 
can decrease while the discriminator loss remains the same, but effectively it can be that the generator image quality
does not improve. Also, a constant discriminator loss can occur if the discriminator is strong initially, and the 
images produced by the generator do not help the discriminator improve further.

"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 128, 
        num_heads = 4,
        num_layers = 6,
        hidden_dim = 32,
        window_size = 16,
        dropout = 0.2,
        lr=0.0001,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers["embed_dim"] = 128
    hypers["num_heads"] = 8
    hypers["num_layers"] = 4
    hypers["hidden_dim"] = 128
    hypers["window_size"] = 128
    hypers["dropout"] = 0.2
    hypers["lr"] = 0.0005

    """
    embed_dim = 256,
    num_heads = 8,
    num_layers = 8,
    hidden_dim = 64,
    window_size = 32,
    dropout = 0.25,
    lr=0.00005,
    """
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**
The first layer does consider only close words in the attention calculation, however after forward pass the values passed to the next level are evaluated from each word and its vicinity. This means that although the next level also considers close indexes, every value already contains information about the context of each word (up to a distances of the window_size). This means that effectively the "contextual receptive field" increases with depth as every value passed on already accounts for the context of the last layers, similar to stacking convolutional layers.
"""

part3_q2 = r"""
**Your answer:**
A proposed solution is to calculate the attention matrix for each word between:    
       (1) the words in distance $w$ from the word     
       (2) the first $w$ words in the sentence     
       (3) the last $w$ words in the sentence    
       This allows calculates every word in context to its neighbours (local-context), the beginning and the end of the sentence. The addition of the beginning and end of the sentence to the attention provides global context because most texts contain the subject statement at the beginning and a summary at the end, which helps bring every word in the sentence with context to the main subject of the text.
Additionally this extra calculation to the attention matrix is still $O(nw)$ because we only calculate an extra $O(w)$ for each word in the sentence.

"""


part4_q1 = r"""
**Your answer:**
In the first part, when we fine-tuned just the final layers, the model average test loss starts by increasing and slightly improves later. This is explained because the model takes a few iterations to fine-tune the last layers to fit the new classification task (while it started from a completely different task), and only after finding the right parameters for sentiment analysis, the model begins to decrease the test loss and "learn". The accuracy slightly decreased between the two epochs, however this might just be chance and given a few more epochs it might improve slightly. What's important is that its accuracy is limited by the features the pretrained back-bone can find, which might not be optimal for this task. This explains why when we fine-tuned the full pretrained model we achieved a way better accuracy - 88% compared to 76%. This let the model both improve the classifier to sentiment analysis task, and the features extracted to focus on the sentiment task at hand.
Compared to Part 3, fine-tuning the full model achieved better accuracy than the trained-from-scratch model, which makes sense because it was deeper and already trained to extract general useful features (like language syntax, context, and meaning). What's interesting to note is that the trained-from-scratch model achieved better accuracy than the fine-tuned-from-last-layers model, which means the pre-trained features on their own weren't optimal for sentiment analysis.
In summary, the best accuracy was achieved by fine-tuning the entire pre-trained model, rather than just the last layers or training from scratch. This result will not always be correct for down-stream tasks (though in most cases it probably will) because some downstream tasks can be very specific and a better accuracy could be achieved by training from scratch for that task, rather than trying to fine-tune a model that was trained to extract non-relevant features for this task. An example could be training a model to understand astrophysics. While llms might know a lot of things they were shown in the training dataset, the information on the topic could be very sparse and it would be better to train a model from scratch only on data relating to the subject.

"""

part4_q2 = r"""
**Your answer:**
Worse. The internal model layer (back-bone) are the feature extractors, that understand the sentence meaning, context, intention etc while the last layers utilize the features extracted by the back-bone to regress the task it was trained on (which might not have been about sentiment!). Therefore, freezing the last layers means the model will utilize the features extracted in the same way, so fine-tuning the back-bone will now tune to be compatible with the last layers, rather than learning new features that are helpful to the new classification task. This means that it will most likely become a worse classifier than training the last layers.
"""


part4_q3= r"""
**Your answer:**
While its not impossible, it isn't effective to use BERT for machine translation because BERT uses an encoder-only architecture and was trained to understand the English language's features and meaning and not to generate new text in a new language. To be able to train it for machine translation it would be way more effective to:     
(1) Change the architecture to encoder-decoder transformer     .  
(2) Modify the text tokenizer to support Language 1 on input tokens and Language 2 on output tokens.     
(3) Pre-train it on **both languages** with the input being a tokenized sentence from Language 1, and the output being the tokenized version of the same sentence, in Language 2.      
This changes will make the model better for machine translation.
"""

part4_q4 = r"""
**Your answer:**
A possible main reason for using RNNs could be memory efficiency. The transformer architecture is limited by the overhead of a maximum sentence length. Specifically, if you want to handle large sentences you need to train the model with a large max-sentence-length, which means the model has to be larger just to be able to handle a few rare cases. This means that transformers need to be very large to handle every case, or be smaller but not work on some cases. RNNs however are flexible and you only need to remember the last model state and continue running through the model as long as you like, without the overhead of extra memory. This allows RNNs to run on low-memory systems and still work for longer sentences.

"""

part4_q5 = r"""
**Your answer:**
NSP (Next Sentence Prediction) is a NLP task where the model is given two sentences (seperated by a special [SEP] character) and is tasked to classify whether the sentences come after each other in the original text they were selected from. The loss is typically Binary Cross Entropy because the labels are 0/1 and we want to compare the model's output probability to the ground-truth. We think that this is a crucial part of pre-training. This task trains the model to understand the relationships between sentences, and how they relate to eachother. It is important that the model understands the meaning of each sentence, but this task also allows it to understand how one sentence affects the other, and allow it to extract long-term context (for many sentences) rather than just understanding each sentence on its own.

"""


# ==============
