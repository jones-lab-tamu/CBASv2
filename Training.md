# Creating a dataset and training a classification model

## 1. Navigate to the Label/Train page in CBASv2.
<p align="left">
    <img src=".//assets/labeltrain.png" alt="CBAS Labeling" style="width: 250px; height: auto;">
</p>
<p align="left"> 

## 2. Hit the plus sign at the bottom of the page and fill out the dataset form.
1. The dataset name should be one word with underscores instead of spaces (e.g. `dataset_1`)
2. Separate behaviors of interest with a semicolon (`;`) with or without a space
   1. Example: `eating; drinking; rearing`
3. Select the camera directories to be included in the dataset. New videos added to these directories will be automatically added to the dataset.
<p align="left">
    <img src=".//assets/createdataset.png" alt="CBAS Dataset" style="width: 250px; height: auto;">
</p>
<p align="left"> 
4. Hit the 'Create' button to finalize the dataset.
<p align="left">
    <img src=".//assets/finalizedataset.png" alt="CBAS Dataset Final" style="width: 250px; height: auto;">
</p>
<p align="left"> 

## 3. Hit the 'Label' button on the dataset to begin labeling videos in the dataset.
<p align="left">
    <img src=".//assets/labelingvids.png" alt="CBAS Labeling" style="width: 525px; height: auto;">
</p>
<p align="left"> 

1. Video surfing:
   1. If you hit the left and right arrows on your keyboard, the video frame will change in those directions (forward and back).
   2. If you hit the up arrow on your keyboard, the 'surf speed' will double (each left and right arrow will go forward or back two frames).
   3. This may still be too slow. Try hitting the up arrow 2-3 times to make the surf speed even faster and move through the video with the left and right arrows.
   4. Hitting the down arrow on your keyboard will halve the surf speed.
   5. If you hold the **Ctrl** key and hit the left or right arrows, CBAS will direct you to another video in the dataset.
2. Labeling a behavior
   1. Your behaviors of interest will be listed on the right side of the screen.
   2. `Code` refers to the keyboard key binding of a particular behavior
   3. `Count` refers to the number of instances and total frames of a behavior that you have labeled.
   4. Surf through the video to find the start of an instance of a behavior you want to label.
   5. Hit the Code of that behavior on your keyboard **once**.
      1. Example: if your first behavior is `eating`, the Code will be `1`. Press 1 on your keyboard.
   6. Surf to the end of the behavior instance (e.g. when the behavior ends).
   7. Hit the Code of that behavior on your keyboard **once** again.
      1. The frames corresponding to that behavior will be colored at the bottom of the video image.
   8. You have successfully labeled an example of a behavior!
   9. If you make an error, press the **Backspace** key.
      1.  This will delete the most recently labeled instance of the behavior.
      2.  The colors corresponding to that behavior at the bottom of the video image will disappear.

## 4. Train the Model
1. Once you have labeled an adequate number of behaviors (a good target is >100 instances per behavior), navigate back to the Label/Train page.
2. On your dataset's card, hit the 'Train' button. This will open a modal with training parameters. The defaults are a good starting point. Click 'Train' in the modal to begin.
3. **Training is a background process.** You will see progress updates in the terminal window that launched CBAS. The application will remain usable during training.
4. After a few moments, the **Precision, Recall, and F1 Score** values will automatically populate in the UI. Performance graphs will also be saved in your project's `data_sets/<your_dataset_name>/` folder.
<p align="left">
    <img src=".//assets/f1.png" alt="CBAS F1" style="width: 250px; height: auto;">
</p>
<p align="left"> 

> **Pro Tip:** How many epochs should you train for? Check the performance graphs (e.g., `f1-score-report.png`) in the dataset folder. If the scores are still improving at the last epoch, you may benefit from training for more epochs. If they have flattened out, the current number is likely sufficient.

## 5. The Inference Workflow: Encode then Classify
After your model is trained, you can use it to classify new, unlabeled videos. This is a two-step background process:

1.  **Encoding:** CBAS must first process your raw videos (`.mp4`) and convert them into feature files (`_cls.h5`). This is a necessary step before classification.
    *   You will see "Encoded..." messages appear in the terminal. This happens automatically for any new videos in your project.
2.  **Inference (Classification):** Once a video is encoded, you can run your trained model on its `.h5` file to get behavior predictions.

> **Important:** You must wait for videos to be **encoded** before you can run **inference** on them.

## 6. Run Inference on Your Videos
1. After your model is trained and the "Encoded..." messages for your target videos have appeared in the terminal, click the **'Infer'** button on your model's card.
2. A modal will appear, showing all recording directories in your project. Check the boxes for the directories containing the videos you want to classify.
3. Click 'Start'. CBAS will use your trained model to analyze the selected videos.
4. This process creates `_outputs.csv` files in the same directories as your videos. These files contain the frame-by-frame prediction probabilities for each behavior, which can then be used in the **Visualize** tab.
<p align="left">
    <img src=".//assets/outputs.png" alt="CBAS outputs" style="width: 610px; height: auto;">
</p>
<p align="left">