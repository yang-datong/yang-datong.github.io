(window.webpackJsonp=window.webpackJsonp||[]).push([["chunk-7cef904f"],{"0bbe":function(e,t,o){"use strict";o("537c")},1071:function(e,t,o){"use strict";o.r(t),o("7f7f");o("57e7"),o("4f37"),o("456d"),o("ac6a"),o("f3e2");var n=o("bd86"),r=o("f564"),a=o("6b41"),n={name:"z-header",data:function(){return{}},computed:{title:function(){return this.$store.getters.titleHeader},showHeader:function(){return this.$store.getters.showHeader},showLeftArrow:function(){return this.$route.meta.canNavBack},showScan:function(){return this.$route.meta.showScan}},methods:{goBack:function(){this.$router.back(-1)},goModuleSelect:function(){this.$router.push({path:"/ModuleSelect"})},scanClick:function(){var o=this;window.cordova&&cordova.plugins.barcodeScanner&&cordova.plugins.barcodeScanner.scan(function(e){var t;e.cancelled||(t="workPerformance-index"===o.$route.name?"/majorhazard/myPerformDuties":"/dw/task/my",o.$router.push({path:t,query:{scanResult:e.text}}))},function(e){Object(r.a)("扫码失败 "+e)},{showTorchButton:!0,prompt:"请扫码单元二维码",resultDisplayDuration:500})}},components:Object(n.a)({},a.a.name,a.a)},a=(o("0bbe"),o("2877")),n=Object(a.a)(n,function(){var e=this,t=e._self._c;return t("div",[t("van-nav-bar",{directives:[{name:"show",rawName:"v-show",value:e.showHeader,expression:"showHeader"}],attrs:{title:e.title,fixed:""},scopedSlots:e._u([{key:"left",fn:function(){return[e.showLeftArrow?t("span",{staticClass:"left-span",on:{click:e.goBack}},[t("van-icon",{attrs:{name:"arrow-left"}}),e._v(" 返回")],1):t("van-icon",{staticClass:"left-span",attrs:{name:"wap-home-o"},on:{click:e.goModuleSelect}})]},proxy:!0},{key:"right",fn:function(){return[e.showScan?t("van-icon",{staticClass:"scan-icon",attrs:{name:"scan",size:"1.2rem"},on:{click:e.scanClick}}):e._e()]},proxy:!0}])})],1)},[],!1,null,"19f6d3d8",null).exports,s=o("3c96"),n={mixins:[o("2a72").a],components:{"z-header":n},data:function(){return{isPwaMode:!0,form:{name:"",id:"",phone:"",locations:"",pcrLocation1:"",pcrLocation2:"",locationcode:"",member1:"",member2:"",member3:"",member4:"",shaiQr:[],shaiQrCode:"",requestForLocationCode:!1}}},created:function(){this.form=Object.assign(this.form,Object(s.a)()),this.form.requestForLocationCode=this.parseBoolean(this.form.requestForLocationCode),StatusBar.styleDefault()},methods:{save:function(){var t=this,o={};Object.keys(this.form).forEach(function(e){"String"==typeof t.form[e]?o[e]=t.form[e].trim():o[e]=t.form[e]}),Object(s.b)(Object.assign({},o)),this.goBack()},parseBoolean:function(e){return 0<=["true","True","YES","yes"].indexOf(e)}}},o=(o("25c7"),Object(a.a)(n,function(){var t=this,e=t._self._c;return e("div",{staticClass:"page"},[e("van-nav-bar",{attrs:{border:!1,title:"配置"},scopedSlots:t._u([{key:"left",fn:function(){return[e("span",{staticClass:"left-span",on:{click:t.goBack}},[e("van-icon",{attrs:{name:"arrow-left",color:"#333",size:"1.5em"}})],1)]},proxy:!0}])}),t._m(0),e("van-form",{ref:"form",on:{submit:t.save}},[e("van-field",{attrs:{label:"姓名"},model:{value:t.form.name,callback:function(e){t.$set(t.form,"name",e)},expression:"form.name"}}),e("van-field",{attrs:{label:"身份证"},model:{value:t.form.id,callback:function(e){t.$set(t.form,"id",e)},expression:"form.id"}}),e("van-field",{attrs:{label:"手机"},model:{value:t.form.phone,callback:function(e){t.$set(t.form,"phone",e)},expression:"form.phone"}}),t.isPwaMode?t._e():e("van-cell",{attrs:{center:"",title:"请求服务器以显示场所码"},scopedSlots:t._u([{key:"right-icon",fn:function(){return[e("van-switch",{attrs:{size:"24"},model:{value:t.form.requestForLocationCode,callback:function(e){t.$set(t.form,"requestForLocationCode",e)},expression:"form.requestForLocationCode"}})]},proxy:!0}],null,!1,524808445)}),e("van-field",{attrs:{label:"场所码地点"},model:{value:t.form.locationcode,callback:function(e){t.$set(t.form,"locationcode",e)},expression:"form.locationcode"}}),e("p",{staticClass:"desc"},[t._v("\n        成员格式：姓名,关系,身份证号\n      ")]),e("van-field",{attrs:{label:"成员1"},model:{value:t.form.member1,callback:function(e){t.$set(t.form,"member1",e)},expression:"form.member1"}}),e("van-field",{attrs:{label:"成员2"},model:{value:t.form.member2,callback:function(e){t.$set(t.form,"member2",e)},expression:"form.member2"}}),e("van-field",{attrs:{label:"成员3"},model:{value:t.form.member3,callback:function(e){t.$set(t.form,"member3",e)},expression:"form.member3"}}),e("van-field",{attrs:{label:"成员4"},model:{value:t.form.member4,callback:function(e){t.$set(t.form,"member4",e)},expression:"form.member4"}})],1),e("van-button",{attrs:{type:"primary"},on:{click:t.save}},[t._v("保存")])],1)},[function(){var e=this,t=e._self._c;return t("p",{staticClass:"desc"},[t("ul",[t("li",[e._v("使用前请先填写姓名与身份证号，填写好信息后，配置页不会再次主动出现。")]),t("li",[e._v("如需修改个人信息，请点击“我的”->“设置“即可返回本页面。")]),t("li",[e._v("扫描场所码会调起相机，3秒后会自动返回无面纱的健康码页面。")]),t("li",[e._v("请提前将手机方向设为“锁定”，避免展示健康码时不慎自动横屏。")])])])}],!1,null,"2029837e",null));t.default=o.exports},"25c7":function(e,t,o){"use strict";o("d851")},"2a72":function(e,t,o){"use strict";t.a={methods:{goBack:function(){var e=this;this.$store.state.goingBack=!0,this.$router.go(-1),setTimeout(function(){e.$store.state.goingBack=!1},500)}}}},"4f37":function(e,t,o){"use strict";o("aa77")("trim",function(e){return function(){return e(this,3)}})},"537c":function(e,t,o){},"57e7":function(e,t,o){"use strict";var n=o("5ca1"),r=o("c366")(!1),a=[].indexOf,s=!!a&&1/[1].indexOf(1,-0)<0;n(n.P+n.F*(s||!o("2f21")(a)),"Array",{indexOf:function(e){return s?a.apply(this,arguments)||0:r(this,e,arguments[1])}})},aa77:function(e,t,o){function n(e,t,o){var n={},r=s(function(){return!!c[e]()||"​"!="​"[e]()}),t=n[e]=r?t(l):c[e];o&&(n[o]=t),a(a.P+a.F*r,"String",n)}var a=o("5ca1"),r=o("be13"),s=o("79e5"),c=o("fdef"),o="["+c+"]",i=RegExp("^"+o+o+"*"),f=RegExp(o+o+"*$"),l=n.trim=function(e,t){return e=String(r(e)),1&t&&(e=e.replace(i,"")),e=2&t?e.replace(f,""):e};e.exports=n},d851:function(e,t,o){},fdef:function(e,t){e.exports="\t\n\v\f\r   ᠎             　\u2028\u2029\ufeff"}}]);