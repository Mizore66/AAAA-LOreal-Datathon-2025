# YouTube Dataset Schema

## Comments Table

| Field | Description |
| :--- | :--- |
| `kind` | Identifies the resource type. The value will be `youtube#comment`. |
| `commentId` | The ID that YouTube uses to uniquely identify the comment. |
| `parentCommentId` | The unique ID of the parent comment. This property is only set if the comment was submitted as a reply to another comment. |
| `channelId` | The ID of the YouTube channel associated with the comment. |
| `videoId` | The ID that YouTube uses to uniquely identify the video. |
| `authorId` | The ID of the user who posted the comment. |
| `textOriginal` | The original, raw text of the comment as it was initially posted or last updated. |
| `likeCount` | The total number of likes (positive ratings) the comment has received. |
| `publishedAt` | The date and time when the comment was originally published (ISO 8601 format). |
| `updatedAt` | The date and time when the comment was last updated (ISO 8601 format). |

## Videos Table

| Field | Description |
| :--- | :--- |
| `kind` | Identifies the resource type. The value will be `youtube#video`. |
| `channelId` | The ID that YouTube uses to uniquely identify the channel that the video was uploaded to. |
| `videoId` | The ID that YouTube uses to uniquely identify the video. |
| `title` | The video’s title. |
| `description` | The video’s description. |
| `tags` | A list of keyword tags associated with the video. Tags may contain spaces (max 500 characters). |
| `defaultLanguage` | The language of the text in the video resource's title and description properties. |
| `defaultAudioLanguage` | The language spoken in the video's default audio track. |
| `contentDuration` | The length of the video in ISO 8601 duration format (e.g., `PT15M33S` for 15 minutes and 33 seconds). |
| `viewCount` | The number of times the video has been viewed. |
| `likeCount` | The number of users who have indicated that they liked the video. |
| `favouriteCount` | The number of users who have indicated that they favorited the video. |
| `commentCount` | The number of comments for the video. |
| `topicCategories` | A list of Wikipedia URLs that provide a high-level description of the video's content. |
| `publishedAt` | The date and time that the video was published. |