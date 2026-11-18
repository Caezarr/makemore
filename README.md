# makemore, part 1: bigram counts (course exercise)

## What this is

A learning log for lecture 2 of Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html): [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo).

This is not original work. The approach and code follow the lecture and Karpathy's reference repository, [karpathy/makemore](https://github.com/karpathy/makemore) (MIT). `names.txt` is the dataset from that repository. I typed the code along with the video and ran it myself.

Status: in progress. The notebook stops at the counting stage, before the lecture turns counts into a model.

## What I implemented

In `makemore_part1_bigrams.ipynb`:

1. Load `names.txt` and inspect it: 32,033 names, shortest 2 characters.
2. Count character bigrams in a Python dict, wrapping each name in `<S>` and `<E>` tokens, then sort by frequency.
3. Rebuild the same counts as a 28x28 `torch.int32` tensor, indexed through `stoi` / `itos` maps (26 letters plus the two tokens).
4. Plot the matrix as an annotated heatmap with matplotlib.

Not done yet: normalising rows into probabilities, sampling names, scoring with negative log-likelihood, and the one-layer neural network trained by gradient descent that the second half of the lecture builds.

## What I learned

- A character-level bigram model is a table of next-character counts. The 32,033 names yield 228,146 bigrams; normalising each row of the table is the whole model.
- Start and end tokens let the same table capture where names begin and end. The most frequent bigrams are `n<E>` (6,763) and `a<E>` (6,640): name endings dominate.
- Using two separate tokens wastes space: `<E>` never starts a bigram and `<S>` never ends one, so a full row and column of the heatmap are zero. The lecture later replaces both with a single `.` token for this reason.
- Moving from a dict keyed by character pairs to a tensor indexed by integers is what makes the later vectorised and neural versions possible.

## Run it

Python 3.10+.

```bash
git clone https://github.com/Caezarr/makemore.git && cd makemore
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook makemore_part1_bigrams.ipynb
```

Run all cells top to bottom; `names.txt` must stay next to the notebook. Outputs are not committed. The dict-counting cell prints every bigram, so its output is long (about 228k lines).

To execute it headless instead:

```bash
jupyter nbconvert --to notebook --execute makemore_part1_bigrams.ipynb --output executed.ipynb
```

This runs in a few seconds on a laptop CPU.

## Credits

All credit for the course, the makemore design, the dataset and the teaching goes to [Andrej Karpathy](https://karpathy.ai). Lecture: <https://www.youtube.com/watch?v=PaCmpygFfXo>. Reference code and data: <https://github.com/karpathy/makemore>.
