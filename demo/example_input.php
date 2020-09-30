<!DOCTYPE HTML>  
<html>
<head>
<style>
.error {color: #FF0000;}
</style>
</head>
<body>  

<?php
header("Content-type:text/html; charset=utf-8");

// define variables and set to empty values (Error Processing)
$title = $content = "";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
  if (empty($_POST["title"])) {
    $title = "";
  } else {
    $title = test_input($_POST["title"]);
  }

  if (empty($_POST["content"])) {
    $content = "";
  } else {
    $content = test_input($_POST["content"]);
  } 
}

function test_input($data) {
  $data = trim($data);
  $data = stripslashes($data);
  $data = htmlspecialchars($data);
  return $data;
}
?>

<h2> AI新聞可信度辨識系統</h2>
<p><span class="error"></span></p>
<form method="post" action="example_result.php">  
  新聞標題 : <textarea name="title" rows="5" cols="40"><?php echo $title;?></textarea>
  <br><br>
  新聞內文 : <textarea name="content" rows="10" cols="80"><?php echo $content;?></textarea>
  <br><br> 
  <input type="submit" name="submit" value="Submit">  
</form>

</body>
</html>
