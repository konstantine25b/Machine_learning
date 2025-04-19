# IEEE-CIS Fraud Detection

Dataset Description
In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.

The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

Categorical Features - Transaction
ProductCD
card1 - card6
addr1, addr2
P_emaildomain
R_emaildomain
M1 - M9
Categorical Features - Identity
DeviceType
DeviceInfo
id_12 - id_38
The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).

You can read more about the data from this post by the competition host.

Files
train_{transaction, identity}.csv - the training set
test_{transaction, identity}.csv - the test set (you must predict the isFraud value for these observations)
sample_submission.csv - a sample submission file in the correct format



კაი დავიწყოთ.

ვფიქრობ რომ რეპოზიტორიის სტუქტურა მექნება ასე ჯერ- პრე პროცესსინგ ცალკე ფაილი სადაც დავამუშავებ მტელ დატას.
და მერე სხვადასხვა მოდელისთვის სხვადასხვა ფაილი და 1 ცალუ model_inference.ipynb საბოლოოდ.

პირობიდან გამომდინარე გვაქვს რამდენიმე კატეგორიული ცვლადი
Categorical Features - Transaction
ProductCD
card1 - card6
addr1, addr2
P_emaildomain
R_emaildomain
M1 - M9
Categorical Features - Identity
DeviceType
DeviceInfo
id_12 - id_38

ასევე გვაქვს ორი ტრეინინგ სეტი ასევე ჩანს რომ TransactionId-ზე უნდა დავაფრედიქთო ფროდია თუ არა.
ეს არის კლასიფიკაციის ამოცანა ამიტომ დავიწყებ ყველაზე მარტივით logistic regression-ის გამოყენებით მარა სანამ დავიწყებ მანამდე ჯერ კაი დიდი პრე პროცესიგი გვაქ გასაკეთებელი 

მარა ვფიქრობ აქამდე რომ ჯერ დატა დავამუშავო რო მერე ცალ ცალკე გავტესტო ყველაზე ამიტო შემქმენი IEEE-CIS Fraud Detection_PreProcessing- სადაც პრე პროცესინგი იქნება.
