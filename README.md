

```python
import scipy.io as spio
import matplotlib.pyplot as plt
```


```python
matdata = spio.loadmat('motor_cortex_data.mat')
```


```python
neuronsData =matdata['spk_rast'][0]
```


```python
def plotNeuron(neuronData,n):
    u,ul,l,dl,d,dr,r,ur=[],[],[],[],[],[],[],[]
    for dat,td in zip(list(neuronData[0]),list(neuronData[1])):#trie_dir
        
        if td[1]>0 and td[0]==0:#up
            u.append(dat)
        elif td[1]<0 and td[0]==0:#down
            d.append(dat)
        elif td[1]==0 and td[0]>0:#right
            r.append(dat)
        elif td[1]==0 and td[0]<0:#left
            l.append(dat)
        elif td[1]>0 and td[0]<0:#up left
            ul.append(dat)
        elif td[1]>0 and td[0]>0:#up right
            ur.append(dat)
        elif td[1]<0 and td[0]<0:#down left
            dl.append(dat)
        elif td[1]<0 and td[0]>0:#down right
            dr.append(dat)
    plt.figure(figsize=(15,10)) 
    
    #up left
    plt.subplot(3, 3, 1)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.eventplot(ul)
    #up 
    plt.subplot(3, 3, 2)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.eventplot(u)
    #up right 
    plt.subplot(3, 3, 3)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.eventplot(ur)

    #left
    plt.subplot(3, 3, 4)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.eventplot(l)
    
    
    #center
    
    plt.subplot(3, 3, 5)
    plt.title('Neurona %d'%n)
    plt.xlim((-1.1,1.1))
    plt.ylim((-1.1,1.1))
    plt.plot([0],[1],'o',label="u")
    plt.plot([0],[-1],'o',label="d")
    plt.plot([-1],[0],'o',label="l")
    plt.plot([1],[0],'o',label="r")
    plt.plot([-.7],[.7],'o',label="ul")
    plt.plot([.7],[.7],'o',label="ur")
    plt.plot([-.7],[-.7],'o',label="dl")
    plt.plot([.7],[-.7],'o',label="dr")
    plt.legend(loc='center')
    
    #right 
    plt.subplot(3, 3, 6)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.eventplot(r)

    #down left
    plt.subplot(3, 3, 7)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.eventplot(dl)
    #down 
    plt.subplot(3, 3, 8)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.eventplot(d)
    #down right 
    plt.subplot(3, 3, 9)
    plt.xlim((0,1))
    plt.ylim((0,20))
    plt.eventplot(dr)
    plt.savefig("neurona%d.png"%i)
    plt.show()
```


```python
#neurona 1 abajo?
#neurona 4 tiene preferencia hacia la Arriba √
#neurona 5 pareciera tener una leve preferencia hacia derecha?
#neurona 7 abajo leve derecha
#neurona 12 tiene preferencia hacia derecha leve arriba
#neurona 13 tiene preferencia abajo leve izquierda
#neurona 15 pareciera tender a la izquierda pero muy difuso
#neurona 17 tiene preferencia izquierda √
#neurona 18 tiene preferencia arriba, leve izquierda
#neurona 21 tiene una muy marcada prefercia hacia abajo
#neurona 24 pareciera tener cierta tendencia hacia abajo derecha
#neurona 26 difuso pero con una cierta tendencia hacia abajo
```


```python
import time
from IPython.display import clear_output

for i in [4,17]:
    #clear_output()
    print "Neurona %d"%i
    plotNeuron(neuronsData[i],i)
    time.sleep(2)
```

    Neurona 4


    /Users/fireness/anaconda2/lib/python2.7/site-packages/numpy/core/_methods.py:29: RuntimeWarning: invalid value encountered in reduce
      return umr_minimum(a, axis, None, out, keepdims)
    /Users/fireness/anaconda2/lib/python2.7/site-packages/numpy/core/_methods.py:26: RuntimeWarning: invalid value encountered in reduce
      return umr_maximum(a, axis, None, out, keepdims)



![png](output_5_2.png)


    Neurona 17



![png](output_5_4.png)



```python
def AnguloPreferencia(neuronData):
    u,ul,l,dl,d,dr,r,ur=[],[],[],[],[],[],[],[]
    for dat,td in zip(list(neuronData[0]),list(neuronData[1])):#trie_dir
         
        if td[1]>0 and td[0]==0:#up
            u.append(len(filter(lambda d:d>0,dat)))
        elif td[1]<0 and td[0]==0:#down
            d.append(len(filter(lambda d:d>0,dat)))
        elif td[1]==0 and td[0]>0:#right
            r.append(len(filter(lambda d:d>0,dat)))
        elif td[1]==0 and td[0]<0:#left
            l.append(len(filter(lambda d:d>0,dat)))
        elif td[1]>0 and td[0]<0:#up left
            ul.append(len(filter(lambda d:d>0,dat)))
        elif td[1]>0 and td[0]>0:#up right
            ur.append(len(filter(lambda d:d>0,dat)))
        elif td[1]<0 and td[0]<0:#down left
            dl.append(len(filter(lambda d:d>0,dat)))
        elif td[1]<0 and td[0]>0:#down right
            dr.append(len(filter(lambda d:d>0,dat)))
    samples = [np.mean(u),np.mean(ul),np.mean(l),np.mean(dl),np.mean(d),np.mean(dr),np.mean(r),np.mean(ur)]
    theta = np.array(np.linspace(np.pi/2,9*np.pi/4,8))
    angles = np.linspace(0,2*np.pi,1000)
    M,m = np.max(sample),np.min(sample)
    amp = M-m
    err = [(np.square((np.cos(theta+a)*amp/2+m+amp/2) - samples)).mean() for a in angles]
    return angles[np.argmin(err)]
```


```python
import numpy as np
def FR(neuronData):
    u,ul,l,dl,d,dr,r,ur=[],[],[],[],[],[],[],[]
    for dat,td in zip(list(neuronData[0]),list(neuronData[1])):#trie_dir
         
        if td[1]>0 and td[0]==0:#up
            u.append(len(filter(lambda d:d>0,dat)))
        elif td[1]<0 and td[0]==0:#down
            d.append(len(filter(lambda d:d>0,dat)))
        elif td[1]==0 and td[0]>0:#right
            r.append(len(filter(lambda d:d>0,dat)))
        elif td[1]==0 and td[0]<0:#left
            l.append(len(filter(lambda d:d>0,dat)))
        elif td[1]>0 and td[0]<0:#up left
            ul.append(len(filter(lambda d:d>0,dat)))
        elif td[1]>0 and td[0]>0:#up right
            ur.append(len(filter(lambda d:d>0,dat)))
        elif td[1]<0 and td[0]<0:#down left
            dl.append(len(filter(lambda d:d>0,dat)))
        elif td[1]<0 and td[0]>0:#down right
            dr.append(len(filter(lambda d:d>0,dat)))
    samples = [np.mean(u),np.mean(ul),np.mean(l),np.mean(dl),np.mean(d),np.mean(dr),np.mean(r),np.mean(ur)]
    return samples
```


```python
FR(neuronsData[1])
```




    [14.55, 13.85, 21.75, 32.2, 36.1, 25.05, 15.1, 15.7]




```python
for i in [4,17]:
    plt.title("Ajuste Coseno Neurona %d"%i)
    sample = FR(neuronsData[i])
    theta = np.array(np.linspace(np.pi/2,9*np.pi/4,8))
    M,m = np.max(sample),np.min(sample)
    amp = M-m
    plt.plot(np.linspace(np.pi/2,9*np.pi/4,8),sample,'ro')
    plt.plot(np.linspace(np.pi/2,9*np.pi/4,1000),np.cos(np.linspace(np.pi/2,9*np.pi/4,1000)+AnguloPreferencia(neuronsData[i]))*amp/2+m+amp/2,'g')
    plt.savefig("CosineTunning%d.png"%i)
    plt.show()
    

```


![png](output_9_0.png)



![png](output_9_1.png)



```python
def cart2pol(x, y):
    """Convert from Cartesian to polar coordinates.

    Example
    -------
    >>> theta, radius = pol2cart(x, y)
    """
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, radius
def compass(u, v, arrowprops=None):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, +0.5, -0.50, -0.90]
    >>> v = [+1, +0.5, -0.45, +0.85]
    >>> compass(u, v)
    """

    angles, radii = cart2pol(u, v)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    kw = dict(arrowstyle="->", color='r')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return fig, ax
```


```python
frn= [np.mean(FR(neuronsData[i])) for i in range(30)]
nrm = np.max(frn)-np.min(frn)
apn= [AnguloPreferencia(neuronsData[i]) for i in range(30)]
u = (np.cos(apn)*frn-np.min(frn))/nrm
v = (np.sin(apn)*frn-np.min(frn))/nrm
fig, ax = compass(u, v)
ax.set_title('Angulos Preferencia segun neuronas\n')

None
```


![png](output_11_0.png)



```python

fig, ax = compass([np.mean(u)], [np.mean(v)],arrowprops={'color':'g'})
ax.set_title('Angulo Preferencia promedio\n')

None
```


![png](output_12_0.png)

