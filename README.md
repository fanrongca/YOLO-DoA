# YOLO-DoA
#### The effectiveness study of YOLO-DoA

| |Methods |Para. |GFlops |MPS |RMSE |
|--- |---  |---  |---    |---    |---    |
|A|YOLO-Basic|22.391M|81.041|1.74|1.9°,6.3°|
|B|YOLO-ResNet18|5.496M|18.649|3.61|1.9°,7.2°|
|C|YOLO-ResNet18<sup>+|0.162M|0.721|8.22|2.3°,7.6°|
|D|+ CSP Connection|0.080M|0.332|8.53|2.2°,7.5°|
|E|+ GIoU Loss|0.080M|0.332|8.23|1.5°,6.5°|
|F|+ SE Operation|0.081M|0.333|8.08|1.4°,6.2°|
|G|+ Grid Sensitive|0.081M|0.333|8.08|1.6°,6.5°|
|H|+ SPP Layer|0.108M|0.397|7.62|1.5°,6.4°|
