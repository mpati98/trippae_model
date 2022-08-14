# Trippae AI model API

- [Trippae AI model API](#trippae-ai-model-api)
  - [Overview](#overview)
  - [Customer segmentation](#customer-segmentation)
    - [Create customer analyzer report](#create-user-analyzer-report)
    - [Clustering customer](#clustering-customer)
  - [Stories segmentation](#stories-segmentation)
    - [Stories with image](#stories-image)
        - [Stories analyzer](#stories-img-analyzer)
        - [Stories clustering](#stories-img-clustering)
    - [Stories with short video](#stories-svd)
        - [Stories analyzer](#stories-svd-analyser)
        - [Stories clustering](#stories-svd-clustering)
  - [Recommendation making](#recommendation-making)
    - [Stories with image](#stories-img-recommend)
    - [Stories with short video](#stories-svideo-recommend)
  - [Update database](#database-update)
    - [Stories with image](#imgdb-update)
    - [Stories with short video](#svideodb-update)

## Overview
- Author: Duang
- Created at: 14/08/2022
- Version: 1.0.0
- Updated at: 
- Status: In progress

### Customer segmentation

```js
GET /customer-segmentation
```

```json
{
    "Description": "grouping customer segments",
    "URL": "https://api.trippae.com/v1/customer-segmentation",
    "Methods": GET,
    "Headers": None
}
```
#### Request BODY:

| Name | Type | Required  | Note |
| ------------- | ------------- | ------------- | ------------- |
| userID  | number  | yes  |  |
| DoB  | datetime  | yes  |  |
| gender  | string  | yes  |  |
| marital status  | string  | yes  |  |
| address  | string  | yes  |  |
| hobies  | string  | yes  |  |
| occupation  | string  | yes  |  |
| online habit   | string  | no  |  |
| interested  | sring  | no  |  |
| usage habits  | string  | no  |  |

#### Response SUCCESS

##### Create customer analyzer report
```js
POST /create-user-analyzer-report
```

##### Clustering customer
```js
POST /clustering-customer
```
| Name | Type | Note  |
| ------------- | ------------- | ------------- |
| userID  | number  |   |
| grouped  | number  | |
| createdAt  | timeskip  ||
| updatedAt  | timeskip  ||

#### Response ERROR
```json
{
    "errors": []
}
```
### Stories segmentation

```js
GET /stories-segmentation
```

```json
{
    "Description": "content analysis, get important feature such as location, negative/positive, hashtag, ...",
    "URL": "https://api.trippae.com/v1/stories-segmentation",
    "Methods": GET,
    "Headers": None
}
```

#### Request BODY:

| Name | Type | Required  | Note |
| ------------- | ------------- | ------------- | ------------- |
| userID | number  | yes  |  |
| storyID | number  | yes  |  |
| content | string  | yes  |  |
| createAt | string  | yes  |  |
| heart | number  | no  | heart of story |
| comment | string  | no  | 10 comments have the most of heart |
| heart | number  | no  | heart of comment |

#### Response SUCCESS

#### Stories with image
##### Stories analyzer
```js
POST /stories-img-analyzer
```
| Name | Type | Note  |
| ------------- | ------------- | ------------- |
| storyID | number |   |
| location | string | |
| attitude | number | 0 - negative, 1 - positive |
| name tags | string | in progress |
| scored | number | |
| createdAt | timeskip | |
| updatedAt | timeskip | |

##### Stories clustering
```js
POST /stories-img-clustering
```
| Name | Type | Note  |
| ------------- | ------------- | ------------- |
| storyID | number |   |
| grouping | number | |
| createdAt  | timeskip | |
| updatedAt  | timeskip | |

#### Stories with short video
##### Stories analyzer
```js
POST /stories-svd-analyzer
```
| Name | Type | Note  |
| ------------- | ------------- | ------------- |
| storyID | number |   |
| location | string | |
| attitude | number | 0 - negative, 1 - positive |
| name tags | string | in progress |
| scored | number | |
| createdAt | timeskip | |
| updatedAt | timeskip | |

##### Stories clustering
```js
POST /stories-svd-clustering
```
| Name | Type | Note  |
| ------------- | ------------- | ------------- |
| storyID | number |   |
| grouping | number | |
| createdAt  | timeskip | |
| updatedAt  | timeskip | |

#### Response ERROR
```json
{
    "errors": []
}
```

### Recommendation making

#### Stories with image
#### Stories with short video

### Update database

#### Stories with image
#### Stories with short video