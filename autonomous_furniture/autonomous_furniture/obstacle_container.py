"""
Global Obstacle Container
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2023-02-28

from typing import Optional

from dynamic_obstacle_avoidance.obstacles import Obstacle

from autonomous_furniture.agent import Furniture, Person


class GlobalObstacleContainer:
    """Singleton-Class of GlobalFurnitureContainer."""

    # _instance: Optional[GlobalFurnitureContainer] = None # Gives me an error (!?)
    _initialized = False
    _instance = None
    # _furniture_list: list[Furniture] = []
    # _obstacle_container: ObstacleContainer = ObstacleContainer()
    _obstacle_list: list[Obstacle] = []

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         print("Create instance of GlobalFurnitureContainer.")
    #         cls._instance = super(GlobalObstacleContainer, cls).__new__(
    #             cls, *args, **kwargs
    #         )
    #     return cls._instance

    @property
    def obstacle_list(self) -> list[Obstacle]:
        return GlobalObstacleContainer._obstacle_list

    @obstacle_list.setter
    def obstacle_list(self, value: list[Obstacle]) -> None:
        GlobalObstacleContainer._obstacle_list = value

    @property
    def initialized(self) -> bool:
        return GlobalObstacleContainer._initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        GlobalObstacleContainer._initialized = value

    def __init__(self, create_new_instance: bool = False) -> None:
        if not create_new_instance and self.initialized:
            return

        if self.initialized:
            print("[INFO] Recreating environment - (careful!)")

        print("[INFO] New environment..")
        self.obstacle_list = []
        # self._instance = True
        self.initialized = True

    def empty(self):
        """Make sure to emtpy it ---
        maybe change this behavior, as it is very error prone (with ipython at least).
        """
        self.obstacle_list = []

    @classmethod
    def get(cls):
        instance = cls(create_new_instance=False)
        return instance

    def append(self, furniture: Furniture) -> None:
        self.obstacle_list.append(furniture)
        print(f"Adding obstacle - total number: {len(self._obstacle_list)}.")

        # self._obstacle_container.append(furniture.get_obstacle_shape())

    def __getitem__(self, key: int) -> Obstacle:
        return self.obstacle_list[key]

    def __setitem__(self, key: int, value: Obstacle) -> None:
        self.obstacle_list[key] = value

    def __len__(self) -> int:
        return len(self.obstacle_list)

    # def get_obstacle_list(self) -> ObstacleContainer:
    #     return self._obstacle_container
