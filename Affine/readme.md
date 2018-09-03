# Affine Analytics ML Challenge
Third Position solution for Affine analytics challenge

## About 
In this competition we were chellenge to make a property recommendation system.The dataset for this competition have following tables:
1. Accounts: This table has information on customers/accounts. These are the accounts for whom they are marketing the properties for sale
2. Opportunities: These include the historic deals for the accounts. Basically, this gives a transaction summary of the deals that have happened between the accounts and the properties. Succesful deal information.
3. Property: This database contains the universal list of properties and its details.
There are two mapping tables are also there:
1. Accounts to Properties: This table comprises information on properties that have been already bought by the accounts.Account information and the property details of the lead
2. Deal to Properties: This table comprises information on the deals that has materialized on the properties.Deal and properties mapping.

Train Having 2727 Accounts and the info about there historic deals.On the otherside, test having 29 new Accounts & no info regarding there Past behaviour.Here Task was to recommend some finite number of properties to these new Accounts.

## Approach
1. Approach was to use Accounts features, Properties features and try to map them in some way.
2. Because there is no history deals information know about the accounts in the testset.we can't apply the Content based Filtering which used the user's historical behaviour and gave suggestions according to that.most suitable in this Cold start situation is the Collaborative Filtering and Hybrid Filtering which used the similarity between customers personal information to gave suggestions.
3. Knn or nearest neighbour were used to find the similar Accounts.then get the properties those most similar customers have bought by using the mapping tables.In the last apply the Knn or nearest neighbour again,This time on the Properties to find the similar properties like that,we will recommend these properties to new Accounts.

## Findings
1. The Properties which were built before 1985 isn't been reccomended to the new users.
2. Properties whose sales year is before 2003 in the Opportunities table in't been recommended to the new users.
3. Properties who having demolished status is also not consider for the reccomendation.so after filtering these properties we were able to remove half of the properties from the Property table.
4. Some Properties have sold more than ones in there lifecycle and sometimes there are more than 1 property sold in a single deal.
