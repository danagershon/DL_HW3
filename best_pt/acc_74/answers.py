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
    hypers = dict(  # TODO LEFT: tweak if have time
        batch_size=512,  # largeer batch size => speed training
        seq_len=64,
        h_dim=128,  # smaller h_dim => speed training but consider 256
        n_layers=2,  # fewer layers => speed training but consider 3
        dropout=0.3,  # consider 0.2
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
        batch_size=64, 
        z_dim=100, 
        discriminator_optimizer={
            'type': 'Adam',
            'lr': 0.0002,
            'betas': (0.5, 0.999)
        }, 
        generator_optimizer={
            'type': 'Adam',
            'lr': 0.0002,
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
    hypers["droupout"] = 0.25
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

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
