<head>

<?php

echo "<span style='font-size:30px'>AI新聞可信度辨識結果<br></span>";
echo "<br>";

echo "loading python... </br>... </br>...";
/*
$command = escapeshellcmd('python3 /home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/demo/demo.py ');
$output = shell_exec($command.$_POST[title].' '.$_POST[content]);
echo $output;
*/

$json_title = json_encode($_POST['title']));
$json_content = json_encode($_POST['content']);

//clear & write in file
$file_handle_title = fopen('title.txt', 'w');
fwrite($file_handle_title, $json_title);
fclose($file_handle_title);

$file_handle_content = fopen('content.txt', 'w');
fwrite($file_handle_content, $json_content);
fclose($file_handle_content);


$path = "python3 /home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/demo/demo_json.py ";
//passthru($path.$param1.' '.$param2);

//echo $param1.' title length: '.strlen($param1);
//echo $param2.' content length: '.strlen($param2);




error_reporting(E_ALL); 
ini_set('display_errors', 1);

exec($path.' 2>&1', $output);

foreach ($output as $value)
    echo $value


?>
<meta http-equiv="Content-Type" content="text/html" charset="utf-8″>
</head>
