# Unfolding Example

A simple Python example demonstrating neutron spectrum unfolding using various methods 
from the literature (e.g. GRAVEL, MAXED).

## Files 

- `unfold.py` - unfolding algorithm 
- `example.py` - a simple runnable example using synthetic data for demonstrative purposes

## Requirements

```bash
pip install -r requirements.txt 
```

## Run 

After installing dependencies, run:

```bash
python example.py
```

This will generate a toy response matrix and detector measurement,
then it employs the unfolding algorithm with a flat prior spectrum and prints the unfolded result

## Implemented methods

- GRAVEL (iterative unfolding method)

Additional methods (e.g. MAXED, Bayesian MLE etc.) may be added in the future.

## Disclaimer

This code is intended for demonstrative purposes only. 
It has not been validated for safety-critical or production use.

## References

- M. Reginatto, *The GRAVEL unfolding algorithm*, Nucl. Instrum. Methods A.
