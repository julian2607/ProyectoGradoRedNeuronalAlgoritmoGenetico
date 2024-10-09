// Alertas
const alertPlaceholder = document.getElementById('liveAlertPlaceholder')

const appendAlert = (message, type) => {
    const wrapper = document.createElement('div')
    wrapper.innerHTML = [
        `<div class="alert alert-${type} alert-dismissible" role="alert">`,
        `   <div>${message}</div>`,
        '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
        '</div>'
    ].join('')
    alertPlaceholder.append(wrapper)
}

// ALERTA
const alertTrigger = document.getElementById('liveAlertBtn')
if (alertTrigger) {
    alertTrigger.addEventListener('click', () => {
        appendAlert('Nice, you triggered this alert message!', 'success')
    })
}

// ACTUALIZAR MODELO
angular.module('myApp', [])
    .controller('myCtrl', function ($scope, $http, $timeout) {
        $scope.control = true;
        function actualizarContador() {
            $scope.control = false;
            fetch('/ActualizarInformacionEntrenamiento')
                .then(response => response.json())
                .then(data => {
                    try {
                        var tablaHtml1 = data.TablaRed;
                        var tablaHtml2 = data.TablaAlgoritmo;
                        document.getElementById('TablaInfoEntrenamiento').innerHTML = tablaHtml1;
                        document.getElementById('TablaInfoEntrenamientoAlgoritmo').innerHTML = tablaHtml2;
                    } catch { }
                });
        }
        setInterval(actualizarContador, 1000);
    });


// VALIDAR BOTON SELECCIONADO
var BotonEntrena = "";
document.getElementById('ENTRENARALGORITMO').addEventListener('click', function () {
    BotonEntrena = "2"
});
document.getElementById('ENTRENARRED').addEventListener('click', function () {
    BotonEntrena = "1"
});

 //MODAL RESULTADO
 var modal = document.getElementById('myModal');
 var modalMessage = document.getElementById('modalMessage');
 var closeModal = document.querySelector('.modal .close');

 //boton cerrar modal de aceptar
 closeModal.addEventListener('click', function() {
    appendAlert('Ha ocurrido un error entrenando el modelo', 'error');
    modal.style.display = 'none';
});

//PETICION ENTRENAR SOLO MODELO
document.getElementById('EntrenarModeloAlgo').addEventListener('submit', function (event) {
    event.preventDefault();
    var loadingScreen = document.getElementById('loadingScreen');
    loadingScreen.classList.remove('hidden');
    var formData = new FormData(this);
    formData.append('Indicador', BotonEntrena);
    fetch('http://127.0.0.1:5000/EntrenarModelo', {
        method: 'POST',
        body: formData
    })        
        .then(data => {
            loadingScreen.classList.add('hidden');
            modal.classList.remove('hidden')
            appendAlert('Proceso realizado con exito', 'success');
            modalMessage.textContent = "Entrenamiento de la red realizada con exito.";
            modal.style.display = 'flex';
            console.log(data);            
        })
        .catch(error => {
            loadingScreen.classList.add('hidden');
            appendAlert('Ha ocurrido un error entrenando el modelo', 'danger');
            console.error('Error:', error);            
        });
});


// //PDF
// document.getElementById("PDF").addEventListener("click", function() {
//     // Redirigir a otra página
//     window.location.href = "http://127.0.0.1:5000/PDF";
// });
// //EXCEL
// document.getElementById("EXCEL").addEventListener("click", function() {
//     // Redirigir a otra página
//     window.location.href = "http://127.0.0.1:5000/EXCEL";
// });
