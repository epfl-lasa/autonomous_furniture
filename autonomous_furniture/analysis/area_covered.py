import numpy as np

class Grid:
    def __init__(self, x_lim, y_lim, furnitures,resolution = [20,20]) -> None:
        self._grid = np.zeros((resolution[0],resolution[1],2))
        self._step_x = (max(x_lim)-min(x_lim))/resolution[1] # x is a column in python 
        self._step_y = (max(y_lim)-min(y_lim))/resolution[0] # Defining the step along x and y to scan the grid / y is a row in python
        self.area_obj_dict = {}

        self.create_grid(x_lim, y_lim, resolution)
        self.create_object_area_struc(furnitures=furnitures, resolution=resolution)

    def create_grid(self, x_lim, y_lim, resolution):
        for ii in range(resolution[0]):
            for jj in range(resolution[1]):
                self._grid[ii,jj] = [(min(y_lim) + ii*self._step_y) + self._step_y/2, (min(x_lim)+ jj*self._step_x + self._step_x/2 ) ] # we store the middle coordinate of each cell in the grid list

    def create_object_area_struc(self, furnitures, resolution):
        for furniture in furnitures:
            self.area_obj_dict[id(furniture)] = ObjectArea(furniture, resolution)

            for ii in range(self._grid.shape[0]):
                for jj in range(self._grid.shape[1]): # Scanning through of each cell center point
                    if furniture.is_inside(self._grid[ii,jj]): # If the point is inside one of the furniture
                        self.area_obj_dict[id(furniture)].old_cell_list[ii,jj] = 255

    def calculate_area(self, furnitures):
        nb_furniture = len(furnitures)

        for ii in range(self._grid.shape[0]):
            for jj in range(self._grid.shape[1]): # Scanning through of each cell center point
                for obj in furnitures:
                    if obj.is_inside(self._grid[ii,jj]): # If the point is inside one of the furniture
                        pass

    def scan(self):
        pass
class ObjectArea:
    def __init__(self, furniture, resolution) -> None:
        self._furniture = furniture
        self.old_cell_list = np.full(resolution, 0)

    def init(self, grid):
        
        pass
        

    # create a function in furniture rectangle to check if the point belongs to rectangle

def main():
    grid=Grid([0,10], [-5,5], [10,20])
    print("hello")

if __name__ == "__main__":
    main()    
