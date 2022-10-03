# tsp-ant-colony

Ant colony optimizer designed to solve Travelling Salesman Problem (TSP). 

# DEMO 

* Input should be in the form of a list [ [x_1, y_1], [x_2, y_2], ... , [x_n, y_n] ]
```python
towns = []

for i in range(60):
    towns.append([randint(0, 100), randint(0, 100)])
```
* How to initialize 
```python
from AntColonyOptimizer import AntColonyOptimizer

ACO_optimizer = AntColonyOptimizer(ants=300, evaporation_rate=0.20, intensification=0.30, alpha=1.00, beta=2.00, beta_evaporation_rate=0.005)
ACO_optimizer.fit(towns, conv_crit=25, mode='min')
```

* Plotting results 
```python
ACO_optimizer.plot()
```
![output_plot_ACO](https://user-images.githubusercontent.com/114445740/193542859-ba24a469-957e-416f-a5de-08230a1b6533.png)

* Visualization 
```python
ACO_optimizer.show_graph(fitted=True)
```
![output_graph_ACO](https://user-images.githubusercontent.com/114445740/193543214-170eadb9-56ee-4a90-a213-a37d6ca3ecb9.png)
