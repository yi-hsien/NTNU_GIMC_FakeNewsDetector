<head>
<?php
echo "<span style='font-size:30px'>AI新聞可信度辨識結果<br></span>";
echo "<br>";
$params1 = $_POST['title'];
$params2 = $_POST['content'];
#echo "$params2";
$trim_params1 = preg_replace('/\s(?=)/', '', $params1);
$trim_params2 = preg_replace('/\s(?=)/', '', $params2);
$path = "python3 /home/csliao/tf01/code/python.py ";
passthru($path.$trim_params1.' '.$trim_params2);
?>
<meta http-equiv="Content-Type" content="text/html" charset="utf-8″>
</head>
