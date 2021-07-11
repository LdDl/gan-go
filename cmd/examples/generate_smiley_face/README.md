## Example for generating smiley face :-)


In binary representation smiley face looks like:
```
0, 1, 1, 0, 0, 0, 1, 1, 0,
0, 1, 1, 0, 0, 0, 1, 1, 0,
0, 1, 1, 0, 0, 0, 1, 1, 0,
0, 1, 1, 0, 0, 0, 1, 1, 0,
0, 0, 0, 0, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 0, 0, 0, 0,
0, 0, 0, 1, 1, 1, 0, 0, 0,
1, 1, 0, 0, 0, 0, 0, 1, 1,
0, 1, 1, 1, 0, 1, 1, 1, 0,
0, 0, 0, 1, 1, 1, 0, 0, 0,
```

If we replace zeros with white spaces and ones with 'X' character then we'll will get:
```
	  x x       x x   
	  x x       x x   
	  x x       x x   
	  x x       x x   
	        x         
	        x         
	      x x x       
	x x           x x 
	  x x x   x x x   
	      x x x       
```

*Note: 1 epoch = 1 training step in this example* 

Generated data on 0-th epoch ():

```
	x   x x     x     
	x x x x x       x 
	  x x x x   x   x 
	  x   x x x     x 
	      x   x x   x 
	x     x x         
	  x         x   x 
	          x   x x 
	  x x     x x x   
	  x             x  
```

Generated data on 200-th epoch:

```
	            x x   
	    x       x x x 
	    x       x x x 
	  x x           x 
	                x 
	        x         
	        x x     x 
	x x           x x 
	    x     x x     
	      x         x    
```

Generated data on last epoch:

*Not exact the same as original image, but it is good enough*
```
	  x x       x x   
	  x x       x x x 
	  x x       x x x 
	  x x       x x x 
	        x       x 
	        x         
	      x x x     x 
	x x           x x 
	  x x x   x x x   
	      x x x     x 
```