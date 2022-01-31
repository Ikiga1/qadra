# Because of scarse ram availablity and bad memory management 
echo "Ready!"
for ((i = 1000; i <= 100000; i+=1000)); do python3 compute_mu.py $i; done