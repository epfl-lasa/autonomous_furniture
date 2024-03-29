import os
from glob import glob
from setuptools import setup

package_name = "autonomous_furniture"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include all launch files
        (os.path.join("share", package_name), glob("launch/*.py")),
        # Include model and simulation files
        (os.path.join("share", package_name), glob("urdf/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anonymous",
    maintainer_email="anonymous@epfl.ch",
    description="TODO: Package description",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "state_publisher = autonomous_furniture.state_publisher:main",
            # 'chair_state_publisher = autonomous_furniture.chair_state_publisher:main',
            # 'table_state_publisher = autonomous_furniture.table_state_publisher:main',
            # 'qolo_state_publisher = autonomous_furniture.qolo_state_publisher:main',
            # "hos_env_publisher = autonomous_furniture.hos_env_publisher.py:main",
        ],
    },
)
