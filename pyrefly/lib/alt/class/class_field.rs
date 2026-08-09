/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Display;
use std::iter;
use std::sync::Arc;

use dupe::Dupe;
use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_graph::index::Idx;
use pyrefly_python::ast::Ast;
use pyrefly_python::dunder;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_types::callable::Callable;
use pyrefly_types::callable::ParamList;
use pyrefly_types::callable::Params;
use pyrefly_types::function::BodyKind;
use pyrefly_types::function::FuncFlags;
use pyrefly_types::function::FunctionKind;
use pyrefly_types::heap::TypeHeap;
use pyrefly_types::quantified::QuantifiedKind;
use pyrefly_types::read_only::IsFinalVariableInitialized;
use pyrefly_types::shaped_array::ShapedArrayType;
use pyrefly_types::simplify::unions;
use pyrefly_types::type_var::PreInferenceVariance;
use pyrefly_types::type_var::Restriction;
use pyrefly_types::typed_dict::TypedDictInner;
use pyrefly_types::types::TParams;
use pyrefly_util::owner::Owner;
use pyrefly_util::prelude::ResultExt;
use pyrefly_util::visit::Visit;
use pyrefly_util::visit::VisitMut;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprTuple;
use ruff_python_ast::helpers::is_dunder;
use ruff_python_ast::name::Name;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::alt::attr::AttrSubsetError;
use crate::alt::attr::ClassBase;
use crate::alt::attr::NoAccessReason;
use crate::alt::callable::CallArg;
use crate::alt::expr::TypeOrExpr;
use crate::alt::types::class_bases::ClassBases;
use crate::alt::types::class_metadata::ClassMetadata;
use crate::alt::types::class_metadata::DataclassKind;
use crate::alt::types::class_metadata::DataclassMetadata;
use crate::alt::types::instance::Instance;
use crate::alt::types::instance::InstanceKind;
use crate::alt::types::pydantic::PydanticModelKind;
use crate::binding::binding::Binding;
use crate::binding::binding::BindingAnnotation;
use crate::binding::binding::ClassFieldDefinition;
use crate::binding::binding::ExprOrBinding;
use crate::binding::binding::KeyAnnotation;
use crate::binding::binding::KeyClassField;
use crate::binding::binding::KeyClassSynthesizedFields;
use crate::binding::binding::MethodSelfKind;
use crate::binding::binding::MethodThatSetsAttr;
use crate::config::error_kind::ErrorKind;
use crate::error::collector::ErrorCollector;
use crate::error::context::ErrorContext;
use crate::error::context::TypeCheckContext;
use crate::error::context::TypeCheckKind;
use crate::error::signature_diff::render_signature_diff;
use crate::solver::solver::SubsetError;
use crate::types::annotation::Annotation;
use crate::types::annotation::Qualifier;
use crate::types::callable::Param;
use crate::types::callable::Required;
use crate::types::class::Class;
use crate::types::class::ClassKind;
use crate::types::class::ClassType;
use crate::types::display::LspDisplayMode;
use crate::types::display::TypeDisplayContext;
use crate::types::function::FuncMetadata;
use crate::types::function::Function;
use crate::types::function::PropertyMetadata;
use crate::types::function::PropertyRole;
use crate::types::keywords::DataclassFieldKeywords;
use crate::types::keywords::TypeMap;
use crate::types::literal::Lit;
use crate::types::quantified::AnchorIndex;
use crate::types::quantified::Quantified;
use crate::types::quantified::QuantifiedIdentity;
use crate::types::quantified::QuantifiedOrigin;
use crate::types::read_only::ReadOnlyReason;
use crate::types::typed_dict::TypedDictField;
use crate::types::types::AnyStyle;
use crate::types::types::BoundMethod;
use crate::types::types::BoundMethodType;
use crate::types::types::CalleeKind;
use crate::types::types::Forall;
use crate::types::types::Forallable;
use crate::types::types::Overload;
use crate::types::types::OverloadType;
use crate::types::types::SuperObj;
use crate::types::types::TArgs;
use crate::types::types::Type;

/// The result of looking up an attribute access on a class (either as an instance or a
/// class access, and possibly through a special case lookup such as a type var with a bound).
#[derive(Debug, Clone)]
pub enum ClassAttribute {
    /// A read-write attribute with a closed form type for both get and set actions.
    ReadWrite(Type),
    /// A read-only attribute with a closed form type for get actions.
    ReadOnly(Type, ReadOnlyReason),
    /// A `NoAccess` attribute indicates that the attribute is well-defined, but does
    /// not allow the access pattern (for example class access on an instance-only attribute)
    NoAccess(NoAccessReason),
    /// A property is a special attribute were regular access invokes a getter.
    /// It optionally might have a setter method; if not, trying to set it is an access error
    Property(Type, Option<Type>, Class),
    /// A descriptor is a user-defined type whose actions may dispatch to special method calls
    /// for the get and set actions.
    Descriptor(Descriptor, DescriptorBase),
}

impl ClassAttribute {
    pub fn read_write(ty: Type) -> Self {
        Self::ReadWrite(ty)
    }

    pub fn read_only(ty: Type, reason: ReadOnlyReason) -> Self {
        Self::ReadOnly(ty, reason)
    }

    pub fn no_access(reason: NoAccessReason) -> Self {
        Self::NoAccess(reason)
    }

    pub fn property(getter: Type, setter: Option<Type>, cls: Class) -> Self {
        Self::Property(getter, setter, cls)
    }

    pub fn descriptor(descriptor: Descriptor, base: DescriptorBase) -> Self {
        Self::Descriptor(descriptor, base)
    }

    pub fn read_only_equivalent(self, reason: ReadOnlyReason) -> Self {
        match self {
            Self::ReadWrite(ty) => Self::ReadOnly(ty, reason),
            Self::Property(getter, _, cls) => Self::Property(getter, None, cls),
            Self::Descriptor(descriptor, base) => Self::Descriptor(
                Descriptor {
                    setter: false,
                    ..descriptor
                },
                base,
            ),
            attr @ (Self::NoAccess(..) | Self::ReadOnly(..)) => attr,
        }
    }

    /// Given a `ClassAttribute`, try to unwrap it as a method type, assuming
    /// that methods are always simple read-only or read-write attributes.
    ///
    /// If we encounter any other case, return `None`.
    pub fn as_instance_method(self) -> Option<Type> {
        match self {
            // TODO(stroxler): ReadWrite attributes are not actually methods but limiting access to
            // ReadOnly breaks unit tests; we should investigate callsites to understand this better.
            ClassAttribute::ReadWrite(ty) | ClassAttribute::ReadOnly(ty, _) => Some(ty),
            ClassAttribute::NoAccess(..)
            | ClassAttribute::Property(..)
            | ClassAttribute::Descriptor(..) => None,
        }
    }

    pub fn is_read_only(&self) -> bool {
        match self {
            ClassAttribute::ReadOnly(_, _)
            | ClassAttribute::Property(_, None, _)
            | ClassAttribute::Descriptor(Descriptor { setter: false, .. }, _) => true,
            _ => false,
        }
    }

    /// Returns true if this attribute represents a data descriptor
    /// (has either `__set__` or `__delete__`), including properties.
    pub fn is_data_descriptor(&self) -> bool {
        match self {
            // All properties are data descriptors: https://docs.python.org/3/howto/descriptor.html#properties.
            ClassAttribute::Property(..) => true,
            // A data descriptor is one that defines `__set__` or `__delete__`:
            // https://docs.python.org/3/reference/datamodel.html#invoking-descriptors
            ClassAttribute::Descriptor(
                Descriptor {
                    setter, deleter, ..
                },
                _,
            ) => *setter || *deleter,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, TypeEq, PartialEq, Eq, Visit, VisitMut)]
pub struct Descriptor {
    /// The location of the property where the descriptor is bound, where we should raise
    /// errors attempting to access the getter/setter.
    range: TextRange,
    /// This is the descriptor class, which is needed both for attribute subtyping
    /// checks in structural types and in the case where there is no getter method.
    cls: ClassType,
    /// Does `__get__` exist on the descriptor?  It is typically a `BoundMethod` although
    /// it is possible for a user to erroneously define a `__get__` with any type, including a
    /// non-callable one.
    getter: bool,
    /// Does `__set__` exist on the descriptor? Similar considerations to `getter` apply.
    setter: bool,
    /// Does `__delete__` exist on the descriptor?
    deleter: bool,
    /// How the descriptor field was initialized. Used to distinguish class-body
    /// descriptors (which have an actual object on the class) from annotation-only
    /// descriptors (which rely on metaclass or other runtime machinery).
    initialization: ClassFieldInitialization,
    /// Whether the defining `def` was decorated with `@override`. Descriptors created by
    /// decorators (e.g. `@classproperty`) become a `ClassType`, which can't carry the
    /// function metadata flag, so we record it here from the undecorated function.
    is_override: bool,
}

#[derive(Clone, Debug)]
pub enum DescriptorBase {
    Instance(ClassType),
    /// Descriptor accessed on a `Self` instance. The `ClassType` is the bounding class,
    /// but the `obj` and `objtype` arguments to `__get__`/`__set__` should use `SelfType`.
    SelfInstance(ClassType),
    ClassDef(ClassBase),
}

/// Correctly analyzing which attributes are visible on class objects, as well
/// as handling method binding correctly, requires distinguishing which fields
/// are assigned values in the class body.
#[derive(Clone, Debug, TypeEq, Visit, VisitMut, PartialEq, Eq)]
pub enum ClassFieldInitialization {
    /// If this is a dataclass field, DataclassFieldKeywords stores the field's
    /// dataclass flags (which are options that control how fields behave).
    ClassBody(Option<Box<DataclassFieldKeywords>>),
    /// This field is initialized in an instance method (e.g. `self.x = 1`).
    ///
    /// Note that this applies only if the field is not declared anywhere else.
    ///
    /// At runtime, this creates an instance attribute. Therefore:
    /// 1. It is not visible when accessed on the class object (e.g. `Class.x` yields a no-access error).
    /// 2. It is ignored by dataclass field extraction (unless also declared in the class body).
    Method,
    /// This field is initialized in a class method (e.g. `cls.x = 1` inside `@classmethod`).
    ///
    /// Note that this applies only if the field is not declared anywhere else.
    ///
    /// At runtime, this creates a class attribute. Therefore:
    /// 1. It is visible when accessed on the class object (e.g. `Class.x` is valid).
    /// 2. It is ignored by dataclass field extraction (unless also declared in the class body).
    ClassMethod,
    /// The field is not initialized at the point where it is declared. At runtime this usually
    /// means the field is instance-only — declared (annotated) but not initialized in the class
    /// body. However, pyrefly intentionally diverges from the runtime here for descriptor
    /// detection: an annotation-only field whose type defines `__get__`/`__set__` is treated as
    /// a class-level descriptor. This is unsound (the descriptor object may never actually be
    /// installed on the class), but matches what other type checkers do and is required for
    /// compatibility with the many library stubs and metaclass-powered patterns (e.g.
    /// `__attributes__`-driven frameworks) that rely on this behavior.
    Uninitialized,
    /// The field is not initialized in the class body or any method in the class,
    /// but we treat it as if it was initialized.
    /// For example, any field defined in a stub file.
    Magic,
}

impl Display for ClassFieldInitialization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClassBody(_) => write!(f, "initialized on class body"),
            Self::Method => write!(f, "initialized in method"),
            Self::ClassMethod => write!(f, "initialized in class method"),
            Self::Uninitialized => write!(f, "initialized on instances"),
            Self::Magic => {
                write!(f, "not initialized on class body/method")
            }
        }
    }
}

impl ClassFieldInitialization {
    fn recursive() -> Self {
        ClassFieldInitialization::ClassBody(None)
    }
}

/// Raw information about an attribute declared somewhere in a class. We need to
/// know whether it is initialized in the class body in order to determine
/// both visibility rules and whether method binding should be performed.
#[derive(Debug, Clone, TypeEq, PartialEq, Eq, Visit, VisitMut)]
pub struct ClassField(ClassFieldInner, IsInherited);

pub enum ClassFieldVariance<'a> {
    Method(&'a Type),
    Property(&'a Type),
    Field { ty: &'a Type, read_only: bool },
}

#[derive(Debug, Clone, TypeEq, PartialEq, Eq, Visit, VisitMut)]
enum ClassFieldInner {
    /// Properties discovered via @property decorator.
    /// Read-onlyness is handled by presence of and type of setter.
    Property { ty: Type, is_abstract: bool },
    /// Descriptors: attributes initialized in the class body whose type has __get__/__set__ methods.
    /// Read-onlyness is handled by descriptor protocol calls.
    Descriptor {
        ty: Type,
        annotation: Option<Annotation>,
        descriptor: Descriptor,
    },
    /// Methods (including abstract methods, functions without return annotations). We always
    /// treat them as read only.
    ///
    /// Callable types are only methods if some form of method binding applies; staticmethods
    /// or Callables that we decide not to model as descriptors become ClassAttributes.
    Method {
        ty: Type,
        is_abstract: bool,
        is_function_without_return_annotation: bool,
    },
    /// A method whose instance attribute type is resolved from another method on the receiver.
    ProxyMethod { target: Name, ty: Type },
    /// Nested class definitions (class statements inside class body).
    /// These are always of type `Type::ClassDef`, and we treat them as read-only.
    NestedClass { ty: Type },
    /// Class attributes (includes staticmethods, Django fields, regular attrs).
    /// These may also be shadowed on instances, unless they are marked as ClassVar.
    ///
    /// To minimize false positives, we treat attributes annotated but not initialized on the
    /// class body as class attributes even though in many cases they will not be defined on
    /// the class; `initialization` tracks information about whether we are sure that access
    /// should succeed.
    ClassAttribute {
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
        read_only_reason: Option<ReadOnlyReason>,
        /// ClassVar: can read from instance, but cannot write/shadow from instance
        is_classvar: bool,
        is_staticmethod: bool,
        /// Django ForeignKey - triggers synthesis of _id field
        is_foreign_key: bool,
        /// Django field with choices - triggers synthesis of get_FOO_display method
        has_choices: bool,
    },
    /// Instance-only attributes (defined in methods, not in class body).
    InstanceAttribute {
        ty: Type,
        annotation: Option<Annotation>,
        read_only_reason: Option<ReadOnlyReason>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProxyMethodAnnotationForm {
    Direct,
    WrappedDirect,
    Other,
}

/// For efficiency, keep track of whether we know from `calculate_class_field`
/// that this is not an inherited field so that we can skip override consistency
/// checks. This information is not needed to understand the class field, it is
/// only used for efficiency.
#[derive(Debug, Clone, TypeEq, PartialEq, Eq, Visit, VisitMut)]
enum IsInherited {
    No,
    Maybe,
}

impl Display for ClassField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ClassFieldInner::Property { ty, .. } => write!(f, "{ty} (property)"),
            ClassFieldInner::Descriptor { ty, .. } => write!(f, "{ty} (descriptor)"),
            ClassFieldInner::Method { ty, .. } => write!(f, "{ty} (method)"),
            ClassFieldInner::ProxyMethod { target, .. } => write!(f, "ProxyMethod[{target}]"),
            ClassFieldInner::NestedClass { ty, .. } => write!(f, "{ty} (nested class)"),
            ClassFieldInner::ClassAttribute {
                ty, initialization, ..
            } => write!(f, "{ty} ({initialization})"),
            ClassFieldInner::InstanceAttribute { ty, .. } => {
                write!(f, "{ty} (instance attribute)")
            }
        }
    }
}

impl ClassField {
    fn new(
        ty: Type,
        annotation: Option<Annotation>,
        initialization: ClassFieldInitialization,
        read_only_reason: Option<ReadOnlyReason>,
        is_foreign_key: bool,
        has_choices: bool,
        is_inherited: IsInherited,
    ) -> Self {
        Self(
            ClassFieldInner::ClassAttribute {
                ty,
                annotation,
                initialization,
                read_only_reason,
                is_classvar: false,
                is_staticmethod: false,
                is_foreign_key,
                has_choices,
            },
            is_inherited,
        )
    }

    pub fn invalid_typed_dict_field(heap: &TypeHeap) -> Self {
        ClassField::new(
            heap.mk_any_error(),
            None,
            ClassFieldInitialization::Magic,
            None,
            false,
            false,
            IsInherited::Maybe,
        )
    }

    pub fn typed_dict_field(
        ty: Type,
        annotation: Annotation,
        read_only_reason: Option<ReadOnlyReason>,
    ) -> Self {
        Self::new(
            ty,
            Some(annotation),
            ClassFieldInitialization::Uninitialized,
            read_only_reason,
            false,
            false,
            IsInherited::Maybe,
        )
    }

    pub fn for_variance_inference(&self) -> (&Type, Option<&Annotation>, bool) {
        match &self.0 {
            ClassFieldInner::Property { ty, .. } => {
                // Properties don't have annotations (defined by decorator)
                (ty, None, self.is_read_only())
            }
            ClassFieldInner::Descriptor { ty, annotation, .. } => {
                // Descriptors may have annotations
                (ty, annotation.as_ref(), self.is_read_only())
            }
            ClassFieldInner::Method { ty, .. } => {
                // Methods don't have annotations and are always read-only
                (ty, None, self.is_read_only())
            }
            ClassFieldInner::ProxyMethod { ty, .. } => (ty, None, self.is_read_only()),
            ClassFieldInner::NestedClass { ty, .. } => (ty, None, self.is_read_only()),
            ClassFieldInner::ClassAttribute { ty, annotation, .. } => {
                (ty, annotation.as_ref(), self.is_read_only())
            }
            ClassFieldInner::InstanceAttribute { ty, annotation, .. } => {
                (ty, annotation.as_ref(), self.is_read_only())
            }
        }
    }

    pub fn variance_inference(&self) -> ClassFieldVariance<'_> {
        match &self.0 {
            ClassFieldInner::Method { ty, .. } => ClassFieldVariance::Method(ty),
       …49826 tokens truncated… DescriptorBase::SelfInstance(class_type) => {
                        let e = NoAccessReason::SettingReadOnlyDescriptor(
                            class_type.class_object().dupe(),
                        );
                        self.error_with_context(
                            errors,
                            range,
                            ErrorKind::ReadOnly,
                            e.to_error_msg(attr_name),
                            context,
                        );
                    }
                    DescriptorBase::ClassDef(_) => {
                        // Class-level assignment bypasses the descriptor protocol.
                        // __set__ only intercepts instance assignments, so we check
                        // that the value is assignable to the descriptor type.
                        let attr_ty = self.heap.mk_class_type(x.cls.clone());
                        self.check_set_read_write_and_infer_narrow(
                            attr_ty,
                            attr_name,
                            got,
                            range,
                            errors,
                            context,
                            false,
                            narrowed_types,
                        );
                    }
                };
                *should_narrow = false;
            }
        }
    }

    pub fn check_class_attr_delete(
        &self,
        class_attr: ClassAttribute,
        attr_name: &Name,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) {
        match class_attr {
            ClassAttribute::NoAccess(reason) => {
                self.error_with_context(
                    errors,
                    range,
                    ErrorKind::NoAccess,
                    reason.to_error_msg(attr_name),
                    context,
                );
            }
            ClassAttribute::ReadOnly(_, reason) => {
                errors
                    .error_builder(
                        range,
                        ErrorKind::ReadOnly,
                        format!("Cannot delete field `{attr_name}`"),
                    )
                    .with_detail(reason.error_message())
                    .emit();
            }
            ClassAttribute::ReadWrite(..)
            | ClassAttribute::Property(..)
            | ClassAttribute::Descriptor(..) => {
                // Allow deleting most attributes for now, for compatibility with mypy.
            }
        }
    }

    /// Filter out overloads from a parent's attribute whose `self` parameter type is
    /// incompatible with the child class. This prevents false positive `bad-override` errors.
    fn filter_overloads_for_override(
        &self,
        attr: ClassAttribute,
        child_cls: &Class,
    ) -> Option<ClassAttribute> {
        let child_type = self
            .heap
            .mk_class_type(self.as_class_type_unchecked(child_cls));
        let filter_type = |ty: Type| -> Option<Type> {
            match ty {
                Type::BoundMethod(bm) if matches!(bm.func, BoundMethodType::Overload(_)) => {
                    // Repeated match because pattern guards cannot move out of bindings.
                    let BoundMethod {
                        obj,
                        func: BoundMethodType::Overload(overload),
                    } = *bm
                    else {
                        unreachable!("guarded by matches! above")
                    };
                    let applicable: Vec<_> = overload
                        .signatures
                        .into_iter()
                        .filter(|sig| {
                            let self_param = match sig {
                                OverloadType::Function(f) => f.signature.get_first_param(),
                                OverloadType::Forall(forall) => {
                                    forall.body.signature.get_first_param()
                                }
                            };
                            self_param.is_none_or(|param| self.is_subset_eq(&child_type, param))
                        })
                        .collect();
                    let signatures = vec1::Vec1::try_from_vec(applicable).ok()?;
                    Some(
                        BoundMethod {
                            obj,
                            func: BoundMethodType::Overload(Overload {
                                signatures,
                                metadata: overload.metadata,
                            }),
                        }
                        .as_type(),
                    )
                }
                other => Some(other),
            }
        };
        match attr {
            ClassAttribute::ReadWrite(ty) => Some(ClassAttribute::ReadWrite(filter_type(ty)?)),
            ClassAttribute::ReadOnly(ty, reason) => {
                Some(ClassAttribute::ReadOnly(filter_type(ty)?, reason))
            }
            other => Some(other),
        }
    }

    pub fn is_class_attribute_subset(
        &self,
        got: &ClassAttribute,
        want: &ClassAttribute,
        is_subset: &mut dyn FnMut(&Type, &Type) -> Result<(), SubsetError>,
    ) -> Result<(), Box<AttrSubsetError>> {
        match (got, want) {
            (_, ClassAttribute::NoAccess(_)) => return Ok(()),
            (ClassAttribute::NoAccess(_), _) => return Err(Box::new(AttrSubsetError::NoAccess)),
            _ => {}
        }
        // Both ClassVar and ClassObjectInitializedOnBody represent class-level read-only
        // attributes, so they are compatible for override purposes.
        let is_classvar_compatible = |attr: &ClassAttribute| {
            matches!(
                attr,
                ClassAttribute::ReadOnly(
                    _,
                    ReadOnlyReason::ClassVar | ReadOnlyReason::ClassObjectInitializedOnBody
                )
            )
        };
        let got_is_classvar = is_classvar_compatible(got);
        let want_is_classvar = is_classvar_compatible(want);
        if got_is_classvar != want_is_classvar {
            return Err(Box::new(AttrSubsetError::ClassVarMismatch {
                got_is_classvar,
            }));
        }
        match (got, want) {
            (_, ClassAttribute::NoAccess(_)) | (ClassAttribute::NoAccess(_), _) => {
                unreachable!("handled above")
            }
            (
                ClassAttribute::Property(_, _, _),
                ClassAttribute::ReadOnly(..) | ClassAttribute::ReadWrite(..),
            ) => Err(Box::new(AttrSubsetError::Property)),
            (
                ClassAttribute::ReadOnly(..),
                ClassAttribute::Property(_, Some(_), _) | ClassAttribute::ReadWrite(_),
            ) => Err(Box::new(AttrSubsetError::ReadOnly)),
            (
                // TODO(stroxler): Investigate this case more: methods should be ReadOnly, but
                // in some cases for unknown reasons they wind up being ReadWrite.
                ClassAttribute::ReadWrite(got),
                ClassAttribute::ReadWrite(want),
            ) if got.has_toplevel_func_metadata() && want.has_toplevel_func_metadata() => {
                is_subset(got, want).map_err(|subset_error| {
                    Box::new(AttrSubsetError::Covariant {
                        got: got.clone(),
                        want: want.clone(),
                        got_is_property: false,
                        want_is_property: false,
                        subset_error,
                    })
                })
            }
            (ClassAttribute::ReadWrite(got), ClassAttribute::ReadWrite(want)) => {
                let subset_error = is_subset(got, want)
                    .map_or_else(Some, |_| is_subset(want, got).map_or_else(Some, |_| None));
                if let Some(subset_error) = subset_error {
                    Err(Box::new(AttrSubsetError::Invariant {
                        got: got.clone(),
                        want: want.clone(),
                        subset_error,
                    }))
                } else {
                    Ok(())
                }
            }
            (
                ClassAttribute::ReadWrite(got) | ClassAttribute::ReadOnly(got, ..),
                ClassAttribute::ReadOnly(want, _),
            ) => is_subset(got, want).map_err(|subset_error| {
                Box::new(AttrSubsetError::Covariant {
                    got: got.clone(),
                    want: want.clone(),
                    got_is_property: false,
                    want_is_property: false,
                    subset_error,
                })
            }),
            (ClassAttribute::ReadOnly(got, _), ClassAttribute::Property(want, _, _)) => {
                is_subset(
                    // Synthesize a getter method
                    &self.heap.mk_callable_ellipsis(got.clone()),
                    want,
                )
                .map_err(|subset_error| {
                    Box::new(AttrSubsetError::Covariant {
                        got: got.clone(),
                        want: want.clone(),
                        got_is_property: false,
                        want_is_property: true,
                        subset_error,
                    })
                })
            }
            (ClassAttribute::ReadWrite(got), ClassAttribute::Property(want, want_setter, _)) => {
                is_subset(
                    // Synthesize a getter method
                    &self.heap.mk_callable_ellipsis(got.clone()),
                    want,
                )
                .map_err(|subset_error| AttrSubsetError::Covariant {
                    got: got.clone(),
                    want: want.clone(),
                    got_is_property: false,
                    want_is_property: true,
                    subset_error,
                })?;
                if let Some(want_setter) = want_setter {
                    // Extract the setter's value param type (the first param
                    // after self) and check setter_param <: got, i.e. the
                    // child's ReadWrite type can accept everything the parent's
                    // setter promises to accept.
                    // Property setters are always a single function (never an
                    // overload), so callable_signatures() returns exactly one.
                    let setter_sigs = want_setter.callable_signatures();
                    if let Some(setter_sig) = setter_sigs.first()
                        && let Some(rest) = setter_sig.strip_first_param()
                        && let Some(setter_value_type) = rest.get_first_param()
                    {
                        is_subset(setter_value_type, got).map_err(|subset_error| {
                            Box::new(AttrSubsetError::Contravariant {
                                want: want_setter.clone(),
                                got: got.clone(),
                                got_is_property: false,
                                want_is_property: true,
                                subset_error,
                            })
                        })
                    } else {
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            }
            (
                ClassAttribute::Property(got_getter, got_setter, _),
                ClassAttribute::Property(want_getter, want_setter, _),
            ) => {
                is_subset(got_getter, want_getter).map_err(|subset_error| {
                    Box::new(AttrSubsetError::Covariant {
                        got: got_getter.clone(),
                        want: want_getter.clone(),
                        got_is_property: true,
                        want_is_property: true,
                        subset_error,
                    })
                })?;
                match (got_setter, want_setter) {
                    (Some(got_setter), Some(want_setter)) => is_subset(got_setter, want_setter)
                        .map_err(|subset_error| {
                            Box::new(AttrSubsetError::Contravariant {
                                want: want_setter.clone(),
                                got: got_setter.clone(),
                                got_is_property: true,
                                want_is_property: true,
                                subset_error,
                            })
                        }),
                    (None, Some(_)) => Err(Box::new(AttrSubsetError::ReadOnly)),
                    (_, None) => Ok(()),
                }
            }
            (
                ClassAttribute::Descriptor(Descriptor { cls: got_cls, .. }, ..),
                ClassAttribute::Descriptor(Descriptor { cls: want_cls, .. }, ..),
            ) => {
                let got_ty = self.heap.mk_class_type(got_cls.clone());
                let want_ty = self.heap.mk_class_type(want_cls.clone());
                is_subset(&got_ty, &want_ty).map_err(|subset_error| {
                    Box::new(AttrSubsetError::Covariant {
                        got: got_ty,
                        want: want_ty,
                        got_is_property: false,
                        want_is_property: false,
                        subset_error,
                    })
                })
            }
            (ClassAttribute::Descriptor(..), _) | (_, ClassAttribute::Descriptor(..)) => {
                Err(Box::new(AttrSubsetError::Descriptor))
            }
        }
    }

    pub fn resolve_get_class_attr(
        &self,
        attr_name: &Name,
        class_attr: ClassAttribute,
        range: TextRange,
        errors: &ErrorCollector,
        context: Option<&dyn Fn() -> ErrorContext>,
    ) -> Result<Type, NoAccessReason> {
        match class_attr {
            ClassAttribute::NoAccess(reason) => Err(reason),
            ClassAttribute::ReadWrite(ty) | ClassAttribute::ReadOnly(ty, _) => Ok(ty),
            ClassAttribute::Property(getter, ..) => {
                self.record_property_getter(range, &getter);
                Ok(self.call_property_getter(getter, range, errors, context))
            }
            ClassAttribute::Descriptor(x, base) => {
                if let Some(getter) = self.resolve_descriptor_getter(attr_name, &x, errors) {
                    // Reading a descriptor with a getter resolves to a method call
                    //
                    // TODO(stroxler): Once we have more complex error traces, it would be good to pass
                    // context down so that errors inside the call can mention that it was a descriptor read.
                    // TODO(mfish33) we allow uninitialized descriptors in this case. This
                    // is to have better compatibility with other type checkers and support
                    // common metaclass patterns. In the future it would be better to detect
                    // this and use an intersection type between the setter and the descriptor
                    // class.
                    Ok(self.call_descriptor_getter(getter, base, range, errors, context))
                } else {
                    // Reading descriptor with no getter resolves to the descriptor itself
                    Ok(self.heap.mk_class_type(x.cls.clone()))
                }
            }
        }
    }

    fn resolve_descriptor_getter(
        &self,
        attr_name: &Name,
        x: &Descriptor,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if x.getter
            && let Some(getter) = self.get_class_member(x.cls.class_object(), &dunder::GET)
        {
            let attr =
                self.as_instance_attribute(&dunder::GET, &getter, &Instance::of_class(&x.cls));
            // `__get__` is bound and called like a method, never re-fed through the
            // descriptor protocol. If it is itself a descriptor, recursing here would
            // loop forever, so report no usable getter.
            if matches!(attr, ClassAttribute::Descriptor(..)) {
                return None;
            }
            Some(
                self.resolve_get_class_attr(attr_name, attr, x.range, errors, None)
                    .unwrap_or_else(|e| {
                        self.error_with_context(
                            errors,
                            x.range,
                            ErrorKind::NoAccess,
                            e.to_error_msg(&dunder::GET),
                            None,
                        )
                    }),
            )
        } else {
            None
        }
    }

    fn resolve_descriptor_setter(
        &self,
        attr_name: &Name,
        x: &Descriptor,
        errors: &ErrorCollector,
    ) -> Option<Type> {
        if x.setter
            && let Some(setter) = self.get_class_member(x.cls.class_object(), &dunder::SET)
        {
            let attr =
                self.as_instance_attribute(&dunder::SET, &setter, &Instance::of_class(&x.cls));
            Some(
                self.resolve_get_class_attr(attr_name, attr, x.range, errors, None)
                    .unwrap_or_else(|e| {
                        self.error_with_context(
                            errors,
                            x.range,
                            ErrorKind::NoAccess,
                            e.to_error_msg(&dunder::SET),
                            None,
                        )
                    }),
            )
        } else {
            None
        }
    }

    /// Return `__call__` as a bound method if instances of `cls` have `__call__`.
    /// This is what the runtime automatically does when we try to call an instance.
    pub fn instance_as_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        if let Some(attr) = self.get_instance_attribute(cls, &dunder::CALL) {
            return self.resolve_dunder_call_attr(attr);
        }
        None
    }

    /// Return `__call__` as bound method when called on `Self`.
    pub fn self_as_dunder_call(&self, cls: &ClassType) -> Option<Type> {
        if let Some(attr) = self.get_self_attribute(cls, &dunder::CALL) {
            return self.resolve_dunder_call_attr(attr);
        }
        None
    }

    /// Return `__call__` as a bound method if instances of `type_var` have `__call__`.
    /// We look up `__call__` from the upper bound of `type_var`, but `Self` is substituted with
    /// the `type_var` instead of the upper bound class.
    pub fn quantified_instance_as_dunder_call(
        &self,
        quantified: Quantified,
        upper_bound: &ClassType,
    ) -> Option<Type> {
        if let Some(attr) =
            self.get_bounded_quantified_attribute(quantified.clone(), upper_bound, &dunder::CALL)
        {
            return self.resolve_dunder_call_attr(attr);
        }
        None
    }

    fn callable_params_and_flags(mut ty: Type) -> Option<(ParamList, FuncFlags)> {
        let mut flags = None;
        ty.transform_toplevel_func_metadata(|meta| {
            if flags.is_none() {
                flags = Some(meta.flags.clone());
            }
        });
        let flags = flags?;
        let params = match ty.callable_signatures().as_slice() {
            [sig] if let Params::List(list) = &sig.params => Some(list.clone()),
            _ => None,
        }?;
        Some((params, flags))
    }

    fn resolve_dunder_call_attr(&self, attr: ClassAttribute) -> Option<Type> {
        let errors = self.error_swallower();
        // The range is only used for error reporting, and the error swallower
        // discards all errors, so a default range is fine here.
        let fake_range = TextRange::default();
        self.resolve_get_class_attr(&dunder::CALL, attr, fake_range, &errors, None)
            .ok()
    }
}
