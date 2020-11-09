
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
?>





<form action="process_bert.php" method="post">

  新聞標題 : <textarea name="title" rows="5" cols="40"><?php header("Content-type:text/html; charset=utf-8"); echo $title;?></textarea>
  <br><br>
  新聞內文 : <textarea name="content" rows="10" cols="80"><?php header("Content-type:text/html; charset=utf-8"); echo $content;?></textarea>
  <br><br> 
  <input type="submit" name="submit" value="Submit">

</form>




</body>
</html>
