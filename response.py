import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

pd.set_option('display.max_columns', 10)

data = pd.read_csv('_28For_Candidate_29_dspm_case_data_set_-__28For_Candidate_29_dspm_case_data_set.csv')

# Part 1

def analyze_cta_performance(df):

    # Added abbreviations to the different cta names, since they were unreadable on the graph
    abbreviations = {
        "Access Your Personalized Mortgage Rates Now": "Access Now",
        "First Time? We've Made it Easy to Find the Best Mortgage Rate" : "First Time?",
        "Get Pre-Approved for a Mortgage in 5 Minutes" : "5 Minutes"
    }

    # Just applying the abbreviations
    data['ctaCopy'] = data['ctaCopy'].map(abbreviations).fillna(data['ctaCopy'])

    # Grouping the data by CTA Copy and Placement
    grouped_data = data.groupby(['ctaCopy', 'ctaPlacement']).agg(
        clicks = ('clickedCTA', 'sum'),
        conversions = ('scheduledAppointment', 'sum'),
        total = ('userId', 'count')
    ).reset_index()

    # Rate Calculations
    grouped_data['click_rate'] = grouped_data['clicks'] / grouped_data['total']
    grouped_data['conversion_rate'] = grouped_data['conversions'] / grouped_data['total']

    # Plotting!
    plt.figure(figsize=(14,7))
    plt.subplot(1,2,1)
    sns.barplot(x = 'ctaCopy', y = 'click_rate', hue = 'ctaPlacement', data = grouped_data)
    plt.title('Click-Through Rates by CTA Copy and Placement')
    plt.xlabel('CTA Copy')
    plt.ylabel('Click Rate')

    plt.subplot(1,2,2)
    sns.barplot(x = 'ctaCopy', y = 'conversion_rate', hue = 'ctaPlacement', data = grouped_data)
    plt.title('Conversion Rates by CTA Copy and Placement')
    plt.xlabel('CTA Copy')
    plt.ylabel('Conversion Rate')

    plt.tight_layout()
    plt.show()

    return grouped_data

# Calling the function and printing the output
cta_performance = analyze_cta_performance(data)
print(cta_performance)

'''
The variables I chose for this analysis are: CTA copy, CTA placement, clicks, total, click rate, and conversion rate.

CTA Copy and Placement: These variables directly influence user interaction, and
different messages and their locations on the page can significantly affect user engagement and actions, 
which are needed to optimize web content for higher conversion rates.

Clicks and Total Impressions: These provide raw data on user engagement. Total impressions count how many 
times the CTA was displayed, while clicks tell us how often the CTA was interacted with. This information 
is fundamental to calculating the click-through rate.

Click-Through Rate: This unit measures the ratio of 
users who click on a specific link to the number of total users who view the page, ad, or any other place the 
link is placed. It helps in understanding which CTA copy and placement grabs more attention.

Conversion Rate: This measures the percentage of visitors who complete a desired action 
(like filling out a form or making a purchase) out of the total number of clicks. High 
conversion rates are indicative of successful content that is relevant and appealing enough 
to persuade users to complete a transaction.

From the data, I can observe distinct patterns:
Higher Engagement with Top Placement: CTAs placed at the top generally perform better in terms of click-through 
and conversion rates compared to those placed in the middle or bottom. This can be attributed to the visibility 
and immediacy of top-placed CTAs. For instance, "5 Minutes" at the top has the highest CTR and conversion rate 
among its placements, suggesting that users are more likely to engage with content that is immediately visible 
when you open the page.

Effectiveness of Specific CTA Copies: The CTA copy "Get Pre-Approved for a Mortgage in 5 Minutes!" consistently 
outperforms other texts across all placements in both CTR and conversion rates, indicating that this particular message 
resonates well with the audience, possibly due to its promise of quick results, which appeals to the audience in a 
time-sensitive context like financial services.
'''

# Part 2

def user_segment_analysis(data):

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    data['deviceType'] = label_encoder.fit_transform(data['deviceType'])
    data['estimatedPropertyType'] = label_encoder.fit_transform(data['estimatedPropertyType'])
    data['ctaCopy'] = label_encoder.fit_transform(data['ctaCopy'])
    data['ctaPlacement'] = label_encoder.fit_transform(data['ctaPlacement'])

    # Select all relevant columns for correlation analysis
    correlation_data = data[['deviceType', 'estimatedAnnualIncome', 'estimatedPropertyType', 'ctaCopy', 'ctaPlacement',
                             'scrollDepth', 'clickedCTA', 'scheduledAppointment']]

    # Compute correlation matrix
    correlation_matrix = correlation_data.corr()

    return correlation_matrix

# Print the matrix
matrix = user_segment_analysis(data)
print(matrix)

'''
The approach for part 2 involves using a detailed correlation analysis to explore the relationships between 
various user attributes—such as device type, estimated annual income, and estimated property type—and their 
interactions with CTAs. This method helps identify which attributes significantly influence user decisions to 
click on a CTA or convert, thus enabling targeted enhancements to the website's content and layout.

Findings

Users accessing the site via different device types might exhibit varying behaviors. For instance, mobile 
users may have different engagement patterns compared to desktop users, possibly due to the device's convenience 
or the context in which it is used (e.g., on-the-go vs. at-home browsing). If data shows that mobile users have a 
lower CTR or conversion rate, it could tell us a need to optimize mobile site design or CTA placement to be more 
mobile-friendly.

Estimated annual income, used to indicate the user's economic demographic, can provide insights into the 
financial products they might be interested in or the likelihood of pursuing higher-value services like mortgages.
A strong correlation between higher income levels and increased engagement or conversion with specific CTAs could 
suggest that more affluent users respond better to certain types of messaging or offers, guiding more personalized 
marketing strategies.

Different property types (residential vs. commercial) might influence user interest in various mortgage products. 
Understanding how users interested in different property types interact with CTAs can help tailor the content and 
offers to better meet their needs. If users looking at commercial properties are less likely to convert, perhaps 
additional information or reassurances are necessary to boost confidence and conversion rates for this segment.

By analyzing which CTAs perform best with specific demographics, Financial Services can refine their CTA copies 
and placements to better align with the preferences and behaviors of distinct user groups. For example, a 
high-performing CTA for younger mobile users might be positioned prominently on mobile landing pages.
Beyond standard demographic and device data, creating interaction features (e.g., device type combined with income 
level) could reveal deeper insights into user behavior patterns, enhancing the predictive power of the model regarding 
CTA performance. Utilizing advanced analytics, such as clustering algorithms to segment users based on their interaction 
patterns, can further refine marketing strategies and content personalization.

In conclusion, the analytical exploration in Part 2 provides vital insights that can significantly influence the 
strategic decisions of Financial Services. By understanding the correlations between user attributes and their 
interactions with CTAs, along with employing sophisticated feature engineering techniques, the company can effectively 
enhance user engagement, optimize conversion rates, and ultimately, drive higher revenue. This data-driven approach 
not only helps in tailoring the user experience to meet diverse needs but also supports the overarching goal of 
advancing users' decision-making journey in the complex landscape of financial services.
'''

# Part 3

def predictive_model(data):

    #label_encoder = LabelEncoder()
    #data['deviceType'] = label_encoder.fit_transform(data['deviceType'])
    #data['estimatedPropertyType'] = label_encoder.fit_transform(data['estimatedPropertyType'])
    #data['ctaCopy'] = label_encoder.fit_transform(data['ctaCopy'])
    #data['ctaPlacement'] = label_encoder.fit_transform(data['ctaPlacement'])

    data = pd.get_dummies(data, columns=['deviceType', 'ctaCopy', 'ctaPlacement'])
    #data['ctaCopy'] = data['ctaCopy'].astype(str)
    #data['ctaPlacement'] = data['ctaPlacement'].astype(str)

    data['ctaCombination'] = data['ctaCopy'] + "_" + data['ctaPlacement']
    #target = label_encoder.fit_transform(data['ctaCombination'])

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(data['ctaCombination'])

    #features = pd.get_dummies(data[['deviceType', 'estimatedAnnualIncome', 'estimatedPropertyType', 'ctaCopy', 'ctaPlacement']])
    features = data.drop(columns=['ctaCombination'])

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    unique, counts = np.unique(predictions, return_counts = True)
    prediction_counts = dict(zip(label_encoder.inverse_transform(unique), counts))
    best_cta = max(prediction_counts, key = prediction_counts.get)

    return report, accuracy, best_cta, prediction_counts[best_cta]

model_report, model_accuracy, best_cta, best_cta_count = predictive_model(data)
print("Model Report:\n", model_report)
print("Model Accuracy:\n", model_accuracy)
print("Best CTA Combination:\n", best_cta)
print("Number of Predictions for Best CTA:\n", best_cta_count)

'''
For this problem, I tried to create a predictive model that encompassed the variables 'deviceType', 'estimatedAnnualIncome',
'ctaCopy', and 'ctaPlacement'. However, I couldn't get the model to work. The configuration looks correct to me, but
there are a ton of different combinations that I tried out (as you can see from the code) but it wouldn't run properly.
Based on the previous results, however, I presume that what would work out the best would most likely be a combination
of the top placement being the best with most users, which would in turn generate a higher revenue!
'''
