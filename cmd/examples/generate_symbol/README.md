## Example for generating symbol

For example I've taken 'H' character.

In binary representation it looks like:
```
0 0 0 0 0 0 0 0 
0 1 1 0 0 1 1 0
0 1 1 0 0 1 1 0
0 1 1 0 0 1 1 0
0 1 1 1 1 1 1 0
0 1 1 1 1 1 1 0
0 1 1 0 0 1 1 0
0 1 1 0 0 1 1 0
0 1 1 0 0 1 1 0
0 0 0 0 0 0 0 0
```

*Note: 1 epoch = 1 training step in this example* 

Generated data on 0-th epoch:

```
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 1
0 0 0 -1 0 0 0 0
0 0 0 -1 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 1 0 0 0
0 0 0 0 0 0 -1 0
0 0 0 0 0 0 0 0
0 0 0 0 -1 0 1 0
```

Generated data on 30-th epoch:

```
-1 0 0 0 0 0 0 0
0 2 1 0 0 2 2 0
0 1 2 0 0 1 2 0
0 1 1 1 0 1 1 0
0 1 1 1 1 1 2 0
0 2 1 2 1 2 2 0
0 1 2 0 0 1 2 0 
0 2 2 0 0 1 1 0
0 2 1 0 0 2 1 0
0 0 0 0 0 0 -1 0
```

Generated data on 150-th epoch:

```
0 0 0 0 0 0 0 0
0 2 2 0 0 1 1 0
0 2 1 0 0 1 2 0
0 1 2 0 0 2 2 0
0 1 1 1 1 1 1 0
0 1 1 1 1 1 1 0
0 2 1 0 0 1 1 0
0 1 1 0 0 1 1 0
0 1 2 0 0 1 1 0
0 0 0 0 0 0 0 0
```

Generated data on last epoch:

```
0 0 0 0 0 0 0 0 
0 1 1 0 0 1 1 0
0 1 1 0 0 1 0 0
0 1 0 0 0 1 1 0
0 1 1 1 1 1 1 0
0 1 1 1 1 1 1 0
0 1 1 0 0 1 1 0
0 1 1 0 0 1 1 0 
0 1 0 0 0 1 1 0
0 0 0 0 0 0 0 0
```