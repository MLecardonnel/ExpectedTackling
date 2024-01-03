# ExpectedTackling

## MOTT: Missed Opportunities To Tackle
### Introduction
The ability to detect missed opportunities in tackling the ball carrier emerges as a critical aspect to work on for the 2024 NFL Big Data Bowl. Traditional statistics often fall short in capturing the nuances of a player's defensive impact, leaving a void in the comprehensive evaluation of a team's defensive performance. The MOTT Metric aims to fill this void by providing a nuanced and insightful perspective on a player's effectiveness in stopping the ball carrier. This metric goes beyond the basic tally of tackles made and missed, delving into the contextual intricacies of each play. By incorporating factors such as player positioning, probability to tackle, and tackling contribution, the MOTT Metric provides a more accurate reflection of a defender's impact on each play. It unveils the instances where a player had a genuine opportunity to make a crucial stop but fell short, shedding light on areas for technique improvement and tactical refinement. In addition to identifying the missed tackles, this metric aims to detect, for example, instances where a defensive player lacks aggression toward the ball carrier or follows a misjudged closing route, resulting in missed opportunities to tackle.

### A Frame-by-Frame Tackling Probability to Detect the Opportunities
To pinpoint instances of missed opportunities to tackle, the initial step involves detecting the tackling opportunities. This identification is achieved through the meticulous creation of a frame-by-frame probability of tackling for each defensive player. The foundation of this probability model is solely based on the positioning and movements of the defensive player, the ball carrier and the blockers. Thus the frame-by-frame features computed for the model are:
- For the defensive player: tracking data (s, a, dis, o, dir), distances and directions to the ball carrier and the three nearest blockers
- For the ball carrier: tracking data (s, a, dis, o, dir), distances to the nearest sideline and to the endzone
- For the three nearest blockers: tracking data (s, a, dis, o, dir)

Orientation and direction features, initially correlated with the play's direction, undergo adjustments to ensure independence from it.

The probability model operates as a binary classification model, where the objective for each defensive player at every frame is to predict whether they will execute a tackle or provide an assist on the ball carrier. The predicted probability computed by the model corresponds to the desired probability of tackling.

 It is anticipated and intended that the model predicts false positives for players during frames where they are in close proximity to stopping the ball carrier. Likewise, false negatives are expected during the initial frames for players who will make a stop at the conclusion of the play. The data exhibits imbalance, given that in a single play, it is probable that none, or only one or two players among the eleven defensive players aligned, will execute a tackle or provide assistance. This aligns perfectly with the intended model, as the mean predicted probability for such data tends to be relatively low. Moreover, specific features play a crucial role in boosting this probability for tackling opportunities.

The model is configured to predict an equivalent proportion of false negatives and false positives and the results are:
<p align="center">
    <img src="reports/figures/confusion_matrix.png">
</p>

The analysis of Shapley values reveals that the model identifies consistent contributions exhibiting clear trends based on the variable modalities. Here are a few examples of features contributions:
<p align="center">
    <img src="reports/figures/contributions_examples.png">
</p>

- The shorter the distance from the defensive player is to the ball carrier, the higher the probability to tackle or assist
- When the ball carrier is situated in the opposite direction from the endzone he aims to score in, relative to the defensive player, it increases the probability of making a tackle or an assist
- A high speed of the defensive player correlates with a higher probability of tackling or providing assistance
- A very short distance from the ball carrier to the endzone reduces the probability to tackle or assist

The tackling probability generated by the model is visualized in the example play below:
<p align="center">
    <img src="reports/animations/animated_play_1.gif">
</p>

It is conceivable that a defensive player may have multiple opportunities to tackle the ball carrier within the same play. Therefore, it is crucial to identify the various opportunities that the defensive player may have during the play. Based on the tackling probability, a new metric is introduced to measure the frame-by-frame opportunity to tackle, the **OTT**. This metric aims to facilitate the detection of tackling opportunities on a play for each defensive player.
$$OTT = \frac{Tackling\ probability}{Distance\ to\ ball\ carrier} \in [0, +\infty[$$

The OTT metric deals with the instability of the frame-by-frame predicted tackling probability and enhances the prominence of the peaks. This simplifies the identification of tackling opportunities thanks to these OTT peaks.

Optimizing the detection of peaks involves conducting a grid search across multiple hyperparameter configurations to pinpoint genuine tackling opportunities. The resulting optimized peak detection yields the following outcomes in this example case of a player having multiple opportunities to tackle within the same play:
<p align="center">
    <img src="reports/figures/peak_detection.png">
</p>

Should no tackling opportunities be identified through this peak detection, the argmax of the OTT marks the frame of the play as the optimal opportunity for the defensive player. This enables the retrieval of smaller yet still significant tackling opportunities to study. Furthermore, it becomes possible to predict the outcome of the best opportunity or opportunities for every defensive player on each play.

### Introducing a New Metric for Missed Opportunities To Tackle
With the optimal tackling opportunities identified for every defensive player on each play, the innovative **MOTT** metric aims to reveal the ones that are not only missed tackles but, to a larger extent, missed opportunities to tackle. The creation of MOTT relies solely on four features for every opportunity:
- The OTT value at the frame of the opportunity
- The average distance between the defensive player and the ball carrier, calculated from the frame of the opportunity to the conclusion of the play
- The distance won by the ball carrier from the frame of the opportunity to the end of the play
- Whether or not the defensive player executes a tackle or provides assistance from the opportunity

The target of the model uses the PFF (Pro Football Focus) scouted and provided metric "pff_missedTackle". It indicates whether or not the defensive player missed a tackle on the play. The target is then a transformation of the PFF metric that denotes whether or not the defensive player missed a tackle for every opportunity. The appropriate model is thus a binary classification model. The objective is to minimize false negatives as much as possible. Given the broader focus on missed tackling opportunities, false positives are also minimized although it is anticipated and intended that some may still be present. These correspond to the missed opportunities that are not missed tackles. Because the chosen feature engineering involves that there is at least one opportunity for every defensive player in each play, the data exhibits a significant imbalance. Thus the model is trained on a sample that reduces the imbalance:
<p align="center">
    <img src="reports/figures/mott_confusion_matrix.png">
</p>

The model achieves great performances. By predicting on the entire dataset, the volume of false positives is higher and corresponds to the missed opportunities that are not missed tackles. The true positives are the well predicted missed tackles. **Collectively, the predicted positives of the model constitute the MOTT**.

The feature contributions instill greater confidence in the model, thanks to the consistent and clear trends observed based on the variable modalities:
<p align="center">
    <img src="reports/figures/mott_contributions_examples.png">
</p>

- The greater the OTT value, the increased likelihood of a MOTT from the defensive player. Conversely, a lower OTT value decreases the probability
- When the distance from the defensive player to the ball carrier remains consistently close from the identified opportunity to the conclusion of the play, the likelihood of a MOTT is very low
- A low distance gained by the ball carrier from the tackling opportunity diminishes the probability of a MOTT for the defensive player
- The likelihood of a MOTT is reduced if the defensive player executes a tackle or provides assistance

The play between the Philadelphia Eagles and the Arizona Cardinals in week five, where DeVonta Smith (Number 6) gained six yards and secured the first down in the third quarter, is selected to showcase **the impressive performance of the MOTT metric on multiple levels**.
<p align="center">
    <img src="reports/animations/video_mott.gif">
</p>

According to NFL Next Gen Stats and PFF data on this play:
- Antonio Hamilton (Number 33) is identified as missing a tackle
- Byron Murphy (Number 7) successfully executes a tackle

However, the model generated three predictions for MOTT on the play:
<p align="center">
    <img src="reports/animations/animated_play_2.gif">
</p>

- The missed tackle by Antonio Hamilton is accurately predicted by the model.
- Despite Byron Murphy being labeled as a tackler in the tracking data, the model predicts his opportunity as a MOTT. Indeed, receiver DeVonta Smith gains an additional five yards and secures the first down after Byron Murphy's attempt at tackling him. It would have been more realistic to consider it as a missed tackle in the provided data.
- Isaiah Simmons (Number 9) has his tackling opportunity predicted as a MOTT. In fact, it is a missed tackling opportunity that is not a missed tackle as he does not make contact with DeVonta Smith. With a more effective closing route to the Eagles receiver or a dive for the legs, Isaiah Simmons could have potentially prevented him from securing the first down.

The model exceeds expectations in its predictions, which diverge from the provided data but appear to align more closely with reality.


### Analysis based on the innovative MOTT metric

Tackling performances can be assessed by comparing the count of MOTT to the total number of tackles and assists for each defensive player. To grasp the subtleties introduced by the metric in the comprehensive evaluation of a defensive player's performance, the same analysis is performed with the tally of missed tackles.

In the figure below, attention is directed towards the tackling performances of safeties, with a particular emphasis on the two safeties from the Jacksonville Jaguars, Rayshawn Jenkins and Andre Cisco.
<p align="center">
    <img src="reports/figures/safeties_tackling_performances.png">
</p>

Rayshawn Jenkins and Andre Cisco both accumulate a similar number of tackles and assists, with respective totals of 48 and 45. However, one could argue that Andre Cisco's tackling performances are significantly better than those of Rayshawn Jenkins, based on missed tackles. Indeed, Rayshawn Jenkins performs much less effectively than the average safety due to his total of 16 missed tackles.

The MOTT metric draws a different conclusion about their tackling performances through its more comprehensive and general approach on tackling opportunities. In the end, Rayshawn Jenkins performs as well as the average safety with a total of 18 MOTT, which is 2 more than his number of missed tackles. Andre Cisco also reaches a total of 18 MOTT, triple the number of his missed tackles. This indicates that Rayshawn Jenkins and Andre Cisco have similar tackling performances in relation to the opportunities they encounter.

Due to the minimal difference between his number of missed tackles and MOTT, it can be inferred that Rayshawn Jenkins has an aggressive profile. He attempts to tackle in every opportunity he gets, even in challenging situations that ultimately result in missed tackles. He, therefore, strives to do everything possible to stop the ball carrier. This could be misinterpreted without the new MOTT metric and because of his high number of missed tackles.

From an offensive standpoint, the introduction of the MOTT metric enables the analysis of avoided tackles, providing additional insights for a comprehensive evaluation of player elusiveness.
<p align="center">
    <img src="reports/figures/offense_elusiveness_performances.png">
</p>

### Conclusion
The MOTT metric introduces an innovative and nuanced approach to evaluating the tackling performances of defensive players in the NFL. The frame-by-frame tackling probability and the introduction of the OTT metric contribute to the accurate identification of genuine tackling opportunities. The final model proves its effectiveness in predicting missed opportunities to tackle, providing insights that closely align with on-field realities compared to conventional data. When combined with other metrics, the MOTT metric provides a more comprehensive evaluation and a deeper understanding of defensive players' performances. Further analysis like reviewing the positioning techniques, field coverage, and tackling methods in instances where players miss opportunities to tackle is instrumental in refining and optimizing their abilities to stop the ball carrier. Moreover, PFF could leverage this metric to pre-select plays featuring potential missed tackles within the missed opportunities, streamlining their analytical processes.
