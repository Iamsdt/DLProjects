Today I create a helper class.

You may notice, most of the projects like image classification you have same types of code, I don't know how you handle all the same code. Actually, I copy and paste all of those codes. To get rid of this problem I create this helper class with the proper documentation for every function.

List of functions:
- prepare_loader() -> for creating date loader
- prepare_loader_split() -> prepare loader by spliting 
- visualize() -> for visualizing data from data loader
- imshow() -> show image file
- load_latest_model() -> load last save model
- save_current_model() -> save current model
- save_check_point() -> save model completely
- load_checkpoint() -> load complete model
- freeze_parameters() -> freeze parameters
- unfreeze() -> unfreeze parameters
- unfreeze_last_layer() -> for unfreezing last layer
- train() -> train model
- train_faster_log() -> train model with log print after certain interval in every epoch.
- check_overfitted() -> check train loss and valid loss
- test_per_class() -> test result per class
- test() -> total test result
- test_with_single_image() -> test with a single image and get model

update methods:



To load this class you can use this command on kaggle or Colab

```!wget https://raw.githubusercontent.com/Iamsdt/DLProjects/master/utils/Helper.py```

Example Notebook

https://www.kaggle.com/iamsdt/kernel6beca45512