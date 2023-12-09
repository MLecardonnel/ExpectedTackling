# ExpectedTackling

## MOTT: Missed Opportunities To Tackle
### Introduction
The ability to detect missed opportunities in tackling the ball carrier emerges as a critical aspect to work on for the 2024 NFL Big Data Bowl. Traditional statistics often fall short in capturing the nuances of a player's defensive impact, leaving a void in the comprehensive evaluation of a team's defensive performance. The MOTT Metric aims to fill this void by providing a nuanced and insightful perspective on a player's effectiveness in stopping the ball carrier. This metric goes beyond the basic tally of tackles made and missed, delving into the contextual intricacies of each play. By incorporating factors such as player positioning, probability to tackle, and tackling contribution, the MOTT Metric provides a more accurate reflection of a defender's impact on each play. It unveils the instances where a player had a genuine opportunity to make a crucial stop but fell short, shedding light on areas for technique improvement and tactical refinement. In addition to identifying the missed tackles, this metric aims to detect, for example, instances where a defensive player lacks aggression toward the ball carrier or follows a misjudged closing route, resulting in missed opportunities to tackle.

### A Frame-by-Frame Tackling Probability to Detect the Opportunities
To pinpoint instances of missed opportunities to tackle, the initial step involves detecting the tackling opportunities. This identification is achieved through the meticulous creation of a frame-by-frame probability of tackling for each defensive player. The foundation of this probability model is solely based on the positioning and movements of the defensive player, the ball carrier and the blockers. Thus the frame-by-frame features computed for the model are:
- For the defensive player: tracking data (s, a, dis, o, dir), distances and directions to the ball carrier and the three nearest blockers
- For the ball carrier: tracking data (s, a, dis, o, dir), distances to nearest sideline and to the endzone
- For the three nearest blockers: tracking data (s, a, dis, o, dir)
Orientation and direction features, initially correlated with the play's direction, undergo adjustments to ensure independence from it.
