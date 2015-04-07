var rest = require('exprestify')
var fs = require('fs')
var exec = require('child_process').exec;

var header = {
    "Access-Control-Allow-Origin": "null",
    "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Credentials": "true"
};

var multiopt = {
    FilePath: "./assets/",
    PostType: "file",
    Rename: function(fld, file) {
        return "CurrentImg";
    }
};

rest.setHeaders(header);

rest.getfile('/image', function(err, query) {
    if (!err) {
        return "./assets/CurrentImg.jpg";
    } else {
        console.log(err);
        return err;
    }
})



rest.multipost('/PostPhoto', function(err, data) {
    if (!err) {
        var pathforPython = __dirname + '/python/py/bin/python ';
        var pathForFile = __dirname + '/python/1.py ';
        exec(pathforPython + pathForFile + __dirname, function(error, stdout, stderr) {
        });
    } else {
        console.log(err);
    }
}, multiopt);

var server = rest.getSocketServer()
var io = require('socket.io')(server)

io.on('connection', function(socket) {
	console.log("Connected: " + socket.id);
    fs.watch('./assets/CurrentImg.jpg', function(event, filename) {
        var time = new Date().getTime();
        socket.emit("ImageModified", "/image?" + time);
    });
});
rest.port = 3000;
server.listen(rest.port, function() {
    console.log("Listening on port 0.0.0.0:%s", rest.port)
})
