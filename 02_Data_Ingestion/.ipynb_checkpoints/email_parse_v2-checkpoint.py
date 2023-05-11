import re
from os import path, listdir
import sys
import json
import pandas as pd
import re
from google.cloud import storage
import io

#
# Precompiled patterns for performance
#
time_pattern = re.compile("Date: (?P<data>[A-Z][a-z]+\, \d{1,2} [A-Z][a-z]+ \d{4} \d{2}\:\d{2}\:\d{2} \-\d{4} \([A-Z]{3}\))")
subject_pattern = re.compile("Subject: (?P<data>.*)")
sender_pattern = re.compile("From: (?P<data>.*)")
recipient_pattern = re.compile("To: (?P<data>.*)")
cc_pattern = re.compile("cc: (?P<data>.*)")
bcc_pattern = re.compile("bcc: (?P<data>.*)")
msg_start_pattern = re.compile("\n\n", re.MULTILINE)
msg_end_pattern = re.compile("\n+.*\n\d+/\d+/\d+ \d+:\d+ [AP]M", re.MULTILINE)

feeds = []
users = {}
threads = {}
thread_users = {}
user_threads = {}

def parse_email(emails,bucket):

    for text in emails:
        #print("prayers")
        #text = TextFile.read().replace("\r", "")
        time = time_pattern.search(text).group("data").replace("\n", "")
        subject = subject_pattern.search(text).group("data").replace("\n", "")

        sender = sender_pattern.search(text).group("data").replace("\n", "")

        recipient = recipient_pattern.search(text).group("data").split(", ")
        cc = cc_pattern.search(text).group("data").split(", ")
        bcc = bcc_pattern.search(text).group("data").split(", ")
        msg_start_iter = msg_start_pattern.search(text).end()
        try:
            msg_end_iter = msg_end_pattern.search(text).start()
            message = text[msg_start_iter:msg_end_iter]
        except AttributeError: # not a reply
            message = text[msg_start_iter:]
        message = re.sub("[\n\r]", " ", message)
        message = re.sub("  +", " ", message)

        # get user and thread ids
        sender_id = get_or_allocate_uid(sender)
        recipient_id = [get_or_allocate_uid(u.replace("\n", "")) for u in recipient if u!=""]
        cc_ids = [get_or_allocate_uid(u.replace("\n", "")) for u in cc if u!=""]
        bcc_ids = [get_or_allocate_uid(u.replace("\n", "")) for u in bcc if u!=""]
        thread_id = get_or_allocate_tid(subject)

        # create a new set if the thread_id does not exist 
        if thread_id not in thread_users:
            thread_users[thread_id] = set()

        # maintain list of users involved in thread
        users_involved = []
        users_involved.append(sender_id)
        users_involved.extend(recipient_id)
        users_involved.extend(cc_ids)
        users_involved.extend(bcc_ids)
        thread_users[thread_id] |= set(users_involved)
        # maintain list of threads where user is involved
        for user in set(users_involved):
            if user not in user_threads:
                user_threads[user] = set()
            user_threads[user].add(thread_id)

        entry =  {"time": time, "subject":subject, "thread": thread_id, "sender": sender_id, "recipient": recipient_id, "cc": cc_ids, "bcc": bcc_ids, "message": message}
        print(entry)
        feeds.append(entry)

    try:
        write_json_to_bucket(feeds,bucket)
        write_json_to_bucket(users,bucket)
        write_json_to_bucket(threads,bucket)
        
        for thread in thread_users:
            thread_users[thread] = list(thread_users[thread])
        write_json_to_bucket(thread_users,bucket)
        
        for user in user_threads:
            user_threads[user] = list(user_threads[user])
        write_json_to_bucket(user_threads,bucket)
        
        # with open('messages_v2.json', 'w') as f:
        #     json.dump(feeds, f)
        # with open('users.json', 'w') as f:
        #     json.dump(users, f)
        # with open('threads.json', 'w') as f:
        #     json.dump(threads, f)
        # with open('thread-users.json', 'w') as f:
        #     for thread in thread_users:
        #         thread_users[thread] = list(thread_users[thread])
        #     json.dump(thread_users, f)
        # with open('user-threads.json', 'w') as f:
        #     for user in user_threads:
        #         user_threads[user] = list(user_threads[user])
        #     json.dump(user_threads, f)
    except IOError:
        print("Unable to write to output files, aborting")
        exit(1)

#
# Function: get_or_allocated_uid
# Arguments: name - string of a user email
# Returns: unique integer id
#
def get_or_allocate_uid(name):
     if name not in users:
         users[name] = len(users)
     return users[name]

def write_json_to_bucket(json_input,bucket):
    # serialize the dictionary as JSON
    json_string = json.dumps(json_input)
    print(json_string)
    # specify the name of the file you want to create in the bucket
    file_name = 'data/'+str(json_input)+'.json'

    # create a Blob object and write the contents to it
    blob = bucket.blob(file_name)
    blob.upload_from_string(json_string)
    
#
# Function: get_or_allocate_tid
# Arguments: name - string of email subject line
# Returns: unique integer id
#
def get_or_allocate_tid(name):
    parsed_name = re.sub("(RE|Re|FWD|Fwd): ", "", name)
    if parsed_name not in threads:
        threads[parsed_name] = len(threads)
    return threads[parsed_name]

def main():
    client = storage.Client()
    bucket = client.get_bucket('user-scripts-msca310019-capstone-49b3')
    
    blob = bucket.blob('data/emails.csv')
    content = blob.download_as_string()
    
    df = pd.read_csv(io.BytesIO(content))
    df = df.sample(100)
    #df = pd.read_csv("/Users/kshitijmittal/Documents/UChicago Acad/Hedwig_AI/Hedwig/00_Data/emails.csv")

    parse_email(list(df.message),bucket)

main()