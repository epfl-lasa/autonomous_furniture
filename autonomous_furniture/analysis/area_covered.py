import numpy as np

class Grid:
    def __init__(self, x_lim, y_lim, resolution = [20,20]) -> None:
        self._grid = np.zeros((resolution[0],resolution[1],2))
        self._step_x = (max(x_lim)-min(x_lim))/resolution[1] # x is a column in python 
        self._step_y = (max(y_lim)-min(y_lim))/resolution[0] # Defining the step along x and y to scan the grid / y is a row in python

        self.create_grid(x_lim, y_lim, resolution)

    def create_grid(self, x_lim, y_lim, resolution):
        for ii in range(resolution[0]):
            for jj in range(resolution[1]):
                self._grid[ii,jj] = [(min(y_lim) + ii*self._step_y) + self._step_y/2, (min(x_lim)+ jj*self._step_x + self._step_x/2 ) ] # we store the middle coordinate of each cell in the grid list

    def calculate_area(self):
        pass

    def scan(self):
        pass
    
    # create a function in furniture rectangle to check if the point belongs to rectangle

def main():
    grid=Grid([0,10], [-5,5], [10,20])
    print("hello")

if __name__ == "__main__":
    main()    
