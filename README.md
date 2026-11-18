# makemore, part 1 (course exercise, in progress)

My work-through of the second lecture of Andrej Karpathy's **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** course: *[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)*.

This is a learning exercise, not original work. The approach and code follow the lecture and Karpathy's reference repository, [karpathy/makemore](https://github.com/karpathy/makemore) (MIT). `names.txt` (about 32k first names) is the dataset from that repository.

## What the notebook covers so far

`makemore_part1_bigrams.ipynb` stops at the bigram counting stage:

1. Load `names.txt` and inspect the dataset (examples, count, word lengths).
2. Count character bigrams with a Python dict, using `<S>` / `<E>` start and end tokens.
3. Build the same counts as a 28×28 `torch` tensor with `stoi` / `itos` character mappings.
4. Plot the count matrix as an annotated heatmap.

Not done yet: turning counts into probabilities, sampling names, measuring negative log-likelihood, and training the equivalent single-layer neural network with gradient descent (the second half of the lecture).

## What I learned

- A character-level bigram model is just a table of next-character counts; normalising each row gives the model.
- Adding explicit start/end tokens lets the same table model where names begin and end.
- Moving from a dict to a tensor indexed through `stoi` / `itos` is what makes the later vectorised (and neural) version possible.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook makemore_part1_bigrams.ipynb
```

Run all cells top to bottom; `names.txt` must stay next to the notebook. Outputs are not committed.

## Credits

All credit for the course, the makemore design, the dataset and the teaching to [Andrej Karpathy](https://karpathy.ai). Lecture: <https://www.youtube.com/watch?v=PaCmpygFfXo>. Reference code and data: <https://github.com/karpathy/makemore>.
