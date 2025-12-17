# NgramLyricGen 🎶
**Author:** [Rajveer Gupta]  
**Project:** Language Modeling using "Never Gonna Give You Up" by Rick Astley

### What is this project?
I wanted to see how a computer could learn to write song lyrics. I used the lyrics from Rick Astley's "Never Gonna Give You Up" as my dataset and compared two different ways to make the computer "learn" the text: first using basic statistical probability (N-gram Histograms) and then using a Deep Learning approach (Neural Networks).

---

### Step 1: The Dataset & Vocab
I started by taking the raw lyrics and tokenizing them (breaking them into individual words). 
* I built a **Vocabulary** which turned every unique word into a numerical index. 
* My total vocabulary size is **81 words**. 
* This mapping is essential because Neural Networks can't process text directly; they need numbers as input.

---

### Step 2: Phase One - The N-gram Histogram
My first attempt was a traditional statistical N-gram model. 
* **The Logic:** I looked at every sequence of 2 words (the context) and counted which word usually follows them in the song. 
* **The Histogram:** I stored these counts in a frequency table. For example, if the sequence "Never gonna" is followed by "give" 10 times, the model assigns a high probability to "give."
* **The Result:** This worked for exact mimicry, but it couldn't "generalize"—it only knows exactly what it has seen in the training data.



---

### Step 3: Phase Two - The Neural Network
After the statistical model, I used **PyTorch** to build a Neural Language Model to see if it could learn the "style" better.

#### The Architecture:
1.  **Embedding Layer:** Instead of just using raw numbers, I turned each word into a 20-dimensional "Embedding." This helps the model learn that certain words are related.
2.  **Flattening:** Since my `Context_Size` is 2, I had two 20-dim vectors. I used `.view()` to flatten them into one 40-dim vector so the Linear layers could read them.
3.  **Hidden Layers:** I used two Linear layers with **ReLU** activations. This is the "brain" of the model where it learns the patterns of the song.
4.  **Output Layer:** The final layer outputs 81 raw scores (logits), one for each word in the vocab.



---

### Step 4: Training & Optimization
* **Loss Function:** I used `CrossEntropyLoss`. I made sure not to put a Softmax inside the model because this function handles it mathematically.
* **Optimizer:** I used `SGD` (Stochastic Gradient Descent) to update the weights during backpropagation.
* **Batch Size:** I set the batch size to 1 to watch the model learn word-by-word.
* **Performance:** My Average Loss reached around **0.99**. This gives a **Perplexity** of about **2.69**, meaning the model is usually narrowed down to about 2 or 3 likely word choices at any time.

---

### Step 5: Text Generation
I wrote a `generate_song_lines` function to actually use the model. 
* It takes a "Seed" (like `["never", "gonna"]`).
* It predicts the next word, adds it to the list, and slides the window forward.
* I added a **Temperature** setting. Lower temperature makes the model very safe/repetitive, while higher temperature makes it more creative and random.

---

### How to run it:
1. Load the lyrics and create the `vocab` using `torchtext`.
2. Define and initialize the `NGramLanguageModel` class.
3. Run the `train()` loop to optimize the weights.
4. Call `generate_song_lines(model, vocab, ["never", "gonna"])` to generate new Astley-style lyrics!

### Key Libraries used:
* `torch`: For building and training the neural network.
* `torchtext`: For vocabulary mapping and text processing.
* `tqdm`: For the training progress bars.
* `math`: To calculate Perplexity from the Loss.