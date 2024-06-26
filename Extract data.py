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
proposed_02_rewards = extract_rewards('./2/events.out.tfevents.1692214720.cv-iits-ra02','std_queue')
proposed_08_rewards = extract_rewards('./8/events.out.tfevents.1691862730.dsw-28692-b46d9b4f8-td46k','std_queue')

# Save the last 30 episodes to CSV
with open('proposed_02_last30_resilience.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(proposed_02_rewards[-30:])

with open('proposed_08_last30_resilience.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(proposed_08_rewards[-30:])
