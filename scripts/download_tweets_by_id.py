from twittercrawler.crawlers import InteractiveCrawler
from twittercrawler.data_io import FileWriter, FileReader
import sys

# load arguments
twitter_api_key_path = sys.argv[1]
seed_file_path = sys.argv[2]
output_path = sys.argv[3]
if len(sys.argv) > 4:
    k = int(sys.argv[4])
else:
    k = 10

# initialize Twitter API
interactive = InteractiveCrawler()
interactive.authenticate(twitter_api_key_path)

# collect tweets by ID
i = 0
not_found = open("not_found.txt", "w")
writer = FileWriter(output_path)
with open(seed_file_path) as fin:
    for line in fin:
        tweet_id = line.rstrip()
        try:
            result = interactive.twitter_api.show_status(id=tweet_id)
            writer.write([result])
        except Exception as e:
            print(e)
            not_found.write("%s\n" % tweet_id)
            print("Error at tweet id:", tweet_id)
        finally:
            i += 1
            if i > k:
                break
writer.close()
interactive.close()
not_found.close()

# reload results and remove duplicates
results_df = FileReader(output_path).read()
results_df = results_df.drop_duplicates(subset=["id_str"])
print(results_df.shape)

# export tweets
results_df.to_csv(output_path.replace(".txt", ".csv"), index=False)
