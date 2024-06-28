import tensorflow as tf
import csv

def extract_rewards(file_path, tag):
    rewards = []
    for summary in tf.compat.v1.train.summary_iterator(file_path):
        for value in summary.summary.value:
            if value.tag == tag:
                rewards.append(value.simple_value)
    return rewards

# Extract rewards
proposed_04_rewards = extract_rewards('./4/events.out.tfevents.cv-iits-ra02','train_reward')

# Save the last 30 episodes to CSV
with open('proposed_02_last30.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(proposed_04_rewards[-30:])

# with open('proposed_08_last30_resilience.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(proposed_08_rewards[-30:])
