# Trippae AI model API

- [Trippae AI model API](#trippae-ai-model-api)
  - [Overview](#overview)
  - [Installations](#installation)
  - [Customer segmentation](#customer-segmentation)
    - [Create customer analyzer report](#create-user-analyzer-report)
    - [Clustering customer](#clustering-customer)
  - [Stories segmentation](#stories-segmentation)
    - [Stories analyzer](#stories-analyzer)
        - [Stories with image](#stories-img-analyzer)
        - [Stories with short video](#stories-svideo-analyzer)
    - [Stories clustering](#stories-clustering)
        - [Stories with image](#stories-img-clustering)
        - [Stories with short video](#stories-svideo-clustering)
  - [Recommendation making](#recommendation-making)
    - [With image stories](#stories-img-recommend)
    - [With short video stories](#stories-svideo-recommend)
  - [Update database](#database-update)
    - [With image stories](#imgdb-update)
    - [With short video stories](#svideodb-update)

## Overview
- Trippae AI Model có vai trò đánh giá các thuộc tính cơ bản của khách hàng nhằm phân loại, phục vụ cho việc gợi ý các stories, video phù hợp với từng phân khúc khách hàng.
- Model được tích hợp vào các API thuận tiện cho việc deploy mô hình lên server.
- Model dựa trên mô hình BERT-NER để phân tích, lọc ra các thuộc tính về địa điểm, khu vực được đề cập đến trong các status, và YOLO v6 trong việc nhận đạng địa diểm xuất hiện trong các short vvideo.


## Installations

## Customer segmentation

```js
POST /customer-segmentation
```

```json
{
    "name": "Vegan Sunshine",
    "description": "Vegan everything! Join us for a healthy breakfast..",
    "startDateTime": "2022-04-08T08:00:00",
    "endDateTime": "2022-04-08T11:00:00",
    "savory": [
        "Oatmeal",
        "Avocado Toast",
        "Omelette",
        "Salad"
    ],
    "Sweet": [
        "Cookie"
    ]
}
```

## Stories segmentation

```js
201 Created
```

```yml
Location: {{host}}/Breakfasts/{{id}}
```

```json
{
    "id": "00000000-0000-0000-0000-000000000000",
    "name": "Vegan Sunshine",
    "description": "Vegan everything! Join us for a healthy breakfast..",
    "startDateTime": "2022-04-08T08:00:00",
    "endDateTime": "2022-04-08T11:00:00",
    "lastModifiedDateTime": "2022-04-06T12:00:00",
    "savory": [
        "Oatmeal",
        "Avocado Toast",
        "Omelette",
        "Salad"
    ],
    "Sweet": [
        "Cookie"
    ]
}
```

## Recommendation making

```js
GET /breakfasts/{{id}}
```

## Update database

```js
200 Ok
```

```json
{
    "id": "00000000-0000-0000-0000-000000000000",
    "name": "Vegan Sunshine",
    "description": "Vegan everything! Join us for a healthy breakfast..",
    "startDateTime": "2022-04-08T08:00:00",
    "endDateTime": "2022-04-08T11:00:00",
    "lastModifiedDateTime": "2022-04-06T12:00:00",
    "savory": [
        "Oatmeal",
        "Avocado Toast",
        "Omelette",
        "Salad"
    ],
    "Sweet": [
        "Cookie"
    ]
}
```
